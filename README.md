# tinyserve：同机双卡 Prefill / Decode 分离

这份 README 从远端环境配置开始，直到两个入口启动、HTTP 请求成功和混合负载评测。
默认使用 **两张 RTX 4090、Linux、1P1D、NCCL、torch attention**。
P、D 各加载完整模型：P 使用 GPU 0，D 使用 GPU 1，HTTP 统一由 P 提供。

项目结构和请求调用链见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)；
调度与显存策略见 [docs/PD_DESIGN.md](docs/PD_DESIGN.md)；
上游原说明保留在 [docs/README_UPSTREAM.md](docs/README_UPSTREAM.md)。

## 1. 把项目放到远端

上传**当前项目的完整代码**，包括新增的 `pd/`、`p_server.py`、`d_server.py`、
`requirements.txt` 和 `scripts/`。只克隆上游仓库不会包含本项目的 PD 扩展。
不要上传 `.venv`；虚拟环境需要在远端重新创建。

例如，在当前 Windows 电脑的 PowerShell 中通过 WSL 打包并上传（替换 `user@SERVER`）：
使用 WSL 的 tar 可以正确处理本机 `.venv` 指向 Linux 目录的符号链接。

```powershell
cd D:\kuroresearch
wsl -d Ubuntu-22.04 -- tar -czf /mnt/d/kuroresearch/tinyserve-pd.tar.gz --exclude=.git --exclude=".venv*" --exclude=.env --exclude=artifacts --exclude=models --exclude=__pycache__ --exclude="pd_smoke*.jsonl" -C /mnt/d/kuroresearch tinyserve
scp .\tinyserve-pd.tar.gz user@SERVER:~/
```

登录远端后解压。以下 Linux 命令均以 `~/tinyserve` 为项目路径：

```bash
tar -xzf ~/tinyserve-pd.tar.gz -C ~
cd ~/tinyserve
ls p_server.py d_server.py requirements.txt scripts/setup_env.sh
```

如果已经通过其他方式上传到这个目录，跳过解压即可。

## 2. 检查远端系统和 GPU

支持 Linux x86_64 / WSL2；推荐 Ubuntu 22.04 或 24.04、Python 3.10–3.12。
本地验证使用 Python 3.10。两个 GPU 进程必须运行在**同一主机、同一 Linux 用户**下。
需联网访问 PyPI、PyTorch 下载源和 Hugging Face，或预先准备对应离线文件。
建议为环境、下载缓存及小模型预留至少 25 GB 磁盘空间。

```bash
uname -sm
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
```

目标机器应显示两张 GPU。安装脚本按驱动分支选择 PyTorch wheel：

| NVIDIA 驱动分支 | 自动选择 | PyTorch |
| --- | --- | --- |
| 580 及以上 | CUDA 13.0 (`cu130`) | 2.14.1 |
| 560–579 | CUDA 12.6 (`cu126`) | 2.14.1 |
| 低于 560 | 停止安装，先升级驱动 | — |

这是本项目的安装预检门槛，最终以实际 CUDA 运算自检为准。两种 wheel 来自
[PyTorch 官方安装源](https://pytorch.org/get-started/previous-versions/)；兼容规则见
[NVIDIA 文档](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)。

若 `nvidia-smi` 不工作，先完成驱动安装。云 GPU 镜像通常已装好；裸机 Ubuntu 可按
[Ubuntu 官方说明](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)
配置后再继续。WSL 使用 Windows 主机的 NVIDIA 驱动。

安装系统 Python 和 venv 组件；如果镜像已经具备这些组件，可跳过：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl ca-certificates
python3 --version
```

## 3. 创建隔离虚拟环境并安装依赖

```bash
cd ~/tinyserve
bash scripts/setup_env.sh
source .venv/bin/activate
python -m pip check
```

脚本会创建项目下的 `.venv`，**不继承系统 site-packages**，从官方 CUDA 源安装
PyTorch，再按 [requirements.txt](requirements.txt) 安装固定版本的直接依赖。
随后检查依赖、模块导入和 GPU BF16 运算，并将实际解析出的全部包版本写入
`artifacts/environment.freeze.txt`。

`datasets` 也在 requirements 中，因为上游 `serve.py` 在导入阶段就依赖它。
默认后端不需要编译 FlashAttention，也不要求单独安装 CUDA Toolkit。
脚本不会安装或升级系统 NVIDIA 驱动。

也支持以下安装选项，**选择一种即可，不需要依次执行**：

```bash
# 显式固定 CUDA 12.6 wheel。
bash scripts/setup_env.sh cu126

# 指定 Python，仅在首次创建环境前选择。
PYTHON_BIN=python3.10 bash scripts/setup_env.sh

# 把环境放到其他目录，例如 WSL 的 Linux 文件系统。
VENV_DIR="$HOME/.venvs/tinyserve-pd" bash scripts/setup_env.sh
source "$HOME/.venvs/tinyserve-pd/bin/activate"
```

后文默认使用项目 `.venv`。重跑脚本会复用已有的隔离环境，不会删除环境，也不会接受
继承系统包的 venv。若提示 `ensurepip is not available`，先安装对应 Python 的 venv 组件。

检查远端的两张卡和项目单元测试：

```bash
export CUDA_VISIBLE_DEVICES=0,1
python scripts/check_env.py --min-gpus 2 --transport nccl
python -m unittest discover -s tests -v
```

自检应输出 `"ok": true` 和两个 GPU 信息，单元测试应全部通过。
自检只检查 CUDA 运算和后端可用性，跨卡 KV 传输由真实模型测试验证。

## 4. 准备模型

先使用已做本地功能验证的 **Qwen2.5-1.5B 基础模型**。以下命令固定 revision，
将模型下载到项目的 `models/` 目录，便于复现实验：

```bash
cd ~/tinyserve
source .venv/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B",
    revision="8faed761d45a263340a0528343f099c05c9a4323",
    local_dir="models/Qwen2.5-1.5B",
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"],
)
PY
export MODEL="$PWD/models/Qwen2.5-1.5B"
ls "$MODEL/config.json"
```

下载使用 [Hugging Face 官方接口](https://huggingface.co/docs/huggingface_hub/guides/download)。
若已有完整模型目录，直接将 `MODEL` 设置为实际路径，跳过下载。
使用其他目录时，也要替换后续检查与 P 启动命令中的模型路径。
目录需要包含模型配置、tokenizer 和全部 safetensors 权重分片。
当前自定义引擎不支持任意模型架构，不能仅凭模型可被 Transformers 加载就认定兼容。

## 5. 首次部署：自动启动两端检查

先执行下面的检查，**此时不要在其他终端手动启动 P/D**：

```bash
cd ~/tinyserve
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export MODEL="$PWD/models/Qwen2.5-1.5B"
PYTHONPATH=. python tests/smoke_launchers.py --model "$MODEL" --devices 0,1 --transport nccl
```

脚本自动启动两个独立入口，检查不同输入长度、HTTP/SSE 输出、取消回收和退出，
再与原始引擎在相同后端下比较 token IDs。成功输出包含以下关键字段：

```json
{"ok": true, "matched_reference": true, "streaming": true, "cancel_reclaimed": true, "clean_shutdown": true}
```

完整结果是 `artifacts/launchers_smoke/summary.json`，P/D 日志和 trace 在同一目录。
检查结束后测试进程自动停止。

**如果双卡 NCCL 不可用**，改用显式 CPU 暂存路径验证：

```bash
PYTHONPATH=. python tests/smoke_launchers.py --model "$MODEL" --devices 0,1 --transport gloo
```

若选择 Gloo，后续 P 启动参数也要使用 `--transport gloo`。
4090 没有 NVLink；使用 NCCL API 本身不证明 KV 走 GPU P2P，实际路径需在目标机器检查。
只有一张卡时可用 `--devices 0,0 --transport gloo` 检查功能，这不代表双卡性能。

## 6. 正式运行：两个终端各启动一个文件

两个终端都在**同一台远端机器**上，使用同一虚拟环境，并设置相同的 GPU 可见范围。
以下示例使用 NCCL；若上一节采用 Gloo，仅修改 P 的传输参数。

**终端 A：启动 P，指定全部策略参数。**

```bash
cd ~/tinyserve
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
python p_server.py \
  --model "$PWD/models/Qwen2.5-1.5B" \
  --devices 0,1 --transport nccl \
  --attention-backend torch \
  --host 0.0.0.0 --port 8000
```

P 显示 `Waiting for d_server.py` 后继续启动 D。`--devices 0,1` 分别指 P、D 使用的卡。

**终端 B：启动 D。**

```bash
cd ~/tinyserve
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
python d_server.py
```

D 自动接收 P 的模型和全部策略配置。两侧引擎加载完成后，P 打印
`Application startup complete`，HTTP 服务才就绪。默认等待连接时间为 300 秒。
D 不开放独立 HTTP 端口，也不需要额外 `torchrun` 或第三个路由服务。

默认策略：页大小 256、prefill chunk 512、batch token 预算 1024、长请求 active
上限 4、headroom 256、efloor 2 页、最大上下文 8192。这些是实验初值。
详细说明见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)，所有参数见
`python p_server.py --help`。

## 7. 发一个请求确认服务

在远端第三个终端执行，不需要激活 Python 环境：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/metrics
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is", "max_tokens":32, "temperature":0, "stream":true}'
```

健康检查返回 `{"status":"ok"}`；生成接口逐个返回 SSE 事件，以 `[DONE]` 结束。
从其他机器访问时，用远端地址替换 `127.0.0.1`，确保 8000 端口可达。
也可以在本机执行 `ssh -L 8000:127.0.0.1:8000 user@SERVER` 建立转发，再访问本机 8000。
服务支持 `/v1/chat/completions`，但聊天接口要求 tokenizer 具有 chat template；
本 README 的基础模型示例使用 `/v1/completions`。

## 8. 运行长短混合负载

保持两端运行，在第三个远端终端执行：

```bash
cd ~/tinyserve
source .venv/bin/activate
python bench_pd.py \
  --model "$PWD/models/Qwen2.5-1.5B" \
  --urls http://127.0.0.1:8000 \
  --requests 1000 --rate 2 --seed 0 \
  --short-tokens 256 --long-tokens 4096 --short-fraction 0.8 \
  --output-tokens 256 --output artifacts/mixed_pd.jsonl
```

这是 80% 短请求、20% 长请求、平均每秒 2 个请求的合成 Poisson 负载。
1000 个请求的派发约需 500 秒，再加排空时间；首次可用 `--requests 20` 检查工具，
但不要据此下 P99 性能结论。终端输出 TTFT / ITL 百分位、吞吐和失败率，逐请求
数据写入指定 JSONL。当前尚未实现 ShareGPT/LongBench 加载器。

### 8.1 对照：请求按到达顺序分配给空闲 KV 页最多的卡

该对照使用两个完整的混合 P+D 实例，每张卡各加载一份模型。CPU 路由器按请求体
接收完成的顺序，逐个查询两端 `/routing_state`，选择 `free_kv_blocks` 最大的实例；
相同时轮转。只串行路由决策，选定后的推理并发执行，非流式请求也不会阻塞后续分配。
请求分配后在该实例完成，不进行跨卡 KV 传输或迁移。

指标是引擎 KV 池中的可分配页数，包含可淘汰的空闲前缀缓存页，不是 `nvidia-smi`
的空闲显存。快照在调度分配页后、forward 前，以及请求完成释放页后发布。
它不预测未来 decode 增长，也不为路由中的请求额外预留 KV；密集到达可能看到相同快照。
两个实例页大小、模型、最大上下文和 attention 后端必须相同。

先停止 P/D，在三个远端终端分别执行。每个终端先进入项目并激活同一虚拟环境：

```bash
cd ~/tinyserve
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export MODEL="$PWD/models/Qwen2.5-1.5B"
```

```bash
# 终端 A：GPU 0 上的完整 P+D 实例。
python baseline_server.py --model "$MODEL" --device 0 --port 8001
```

```bash
# 终端 B：GPU 1 上的完整 P+D 实例。
python baseline_server.py --model "$MODEL" --device 1 --port 8002
```

```bash
# 终端 C：CPU 路由器，不占用额外 GPU。
python baseline_router.py --backends http://127.0.0.1:8001,http://127.0.0.1:8002 \
  --policy most_free_kv --port 9000
```

第四个终端在相同项目环境中执行：

```bash
curl --fail http://127.0.0.1:9000/health
python bench_pd.py --model "$PWD/models/Qwen2.5-1.5B" \
  --urls http://127.0.0.1:9000 \
  --warmup-urls http://127.0.0.1:8001,http://127.0.0.1:8002 \
  --requests 1000 --rate 2 --seed 0 \
  --short-tokens 256 --long-tokens 4096 --short-fraction 0.8 \
  --output-tokens 256 --output artifacts/mixed_free_kv.jsonl
```

`--warmup-urls` 确保每个副本都预热。TTFT 从客户端发送开始计时，包含排队、KV 查询、
代理和模型计算的成本；P99 ITL 使用客户端收到的 token 间隔。
逐请求结果保存选中的副本编号、路由序号和分配时两端空闲页数。
路由器 `/metrics` 提供各副本分配数、活动请求和最近 256 次分配；调试时可加
`--trace artifacts/routing.jsonl` 保存全部分配记录，纯性能测试默认关闭 trace。
KV 状态查询失败时该次请求返回错误；转发失败不自动重试。

### 8.2 三组实验的比较方式

| 组别 | GPU 工作方式 | 路由器参数 |
| --- | --- | --- |
| Mixed + Round Robin | 两张卡各执行完整 P+D | 两个基线 URL，`--policy round_robin` |
| Mixed + Most Free KV | 两张卡各执行完整 P+D | 两个基线 URL，`--policy most_free_kv` |
| PD | GPU 0 prefill、GPU 1 decode | P 的单个 URL，`--policy round_robin` |

为使三组都经过同一层 HTTP 代理，PD 组也可以在启动 P/D 后运行：

```bash
# 先停止上一组路由器及两个基线实例，再按第 6 节启动 P/D。
python baseline_router.py --backends http://127.0.0.1:8000 --policy round_robin --port 9000
```

PD 的 bench 同样使用 `--urls http://127.0.0.1:9000`，预热地址改为
`--warmup-urls http://127.0.0.1:8000`，其他负载参数保持一致。直接访问 P 的结果与
经过代理的结果应分别标注。每组开始前重启模型实例清空上一轮前缀缓存，再预热。
每组固定相同 GPU 数量、模型、精度、attention 后端、页大小、上下文和负载种子。

这是完整 serving 方案的对照：混合实例保留上游混合调度器，PD 使用本项目的长短队列
和预留/扩容策略。若要单独归因 PD 分离，需要再做调度和显存策略一致的消融实验。

新增对照的正确性检查（自动启动两个基线实例和路由器，检查完成后自动退出）：

```bash
PYTHONPATH=. python tests/smoke_routing.py --model "$PWD/models/Qwen2.5-1.5B" --devices 0,1
# 单卡功能检查使用 --devices 0,0；不能据此比较双卡性能。
```

## 9. 停止、重启与排查

正常停止：先在 P 终端按 Ctrl+C，P 通知 D 退出并清理控制 socket。
单独停止 D 会使 P 的健康检查返回 503。通信器或 worker 故障后停止两端再重启。
修改传输方式或策略参数也需要重启两端。

| 现象 | 处理 |
| --- | --- |
| `No module named ...` | 激活 `.venv` 后运行 `python -m pip check`；重跑安装脚本补齐依赖 |
| `CUDA unavailable` | 检查 `nvidia-smi`、`CUDA_VISIBLE_DEVICES` 和 CUDA wheel；不能使用 CPU 版 torch |
| 只看到一张 GPU | 两端都设置 `CUDA_VISIBLE_DEVICES=0,1`，执行 `check_env.py --min-gpus 2` |
| P 一直等待 D | 确认 D 已启动、同机同用户且 `--ipc-dir` 相同；默认 `/tmp/tinyserve-pd-<UID>` |
| `Another P server already owns ...` | 停止原实例，或为两端设置新的 `--ipc-dir` 并更换 HTTP 端口 |
| NCCL 报错/超时 | 停止两端，将 P 改为 `--transport gloo`，重新跑对应 smoke test |
| 显存不足 | 用 1.5B 小模型起步，停掉其他 GPU 任务，降低上下文长度、并发数或显存利用率配置 |
| 权重下载失败 | 检查网络，或上传完整模型目录并修改 P 的路径 |
| `/health` 返回 503 | 查看两端日志；D 退出或 worker/通信错误时需两端一起重启 |

| 文件 | 作用 |
| --- | --- |
| `requirements.txt` | 直接依赖版本 |
| `scripts/setup_env.sh` | 创建隔离 venv、选择 CUDA wheel、安装、自检 |
| `scripts/check_env.py` | 检查 imports、GPU 数量、BF16 和通信后端 |
| `artifacts/environment.freeze.txt` | 本次实际安装的全部包版本 |
| `tests/smoke_launchers.py` | 两个独立入口的真实模型端到端验证 |

验证范围：本机只有一张 4090 Laptop，可做单卡双进程 Gloo 功能验证；
双卡 NCCL、实际通信路径与 TTFT / P99 ITL 需要在远端实测。

2026-11-25 已从隔离 venv 完整安装并验证：Python 3.10.12、PyTorch 2.14.1+cu130、
`pip check` 和 22 项单元测试通过；Qwen2.5-1.5B 的独立 P/D HTTP/SSE、取消回收、
退出与参考 token IDs 比较通过。本机结果在 `artifacts/fresh_env_smoke/summary.json`。

2026-11-26 新增可用 KV 页路由对照后，29 项单元测试通过；Qwen2.5-1.5B
单卡双实例验证了按最大空闲页分配、FIFO 路由决策、流式转发、取消回收以及
评测脚本记录路由信息，生成 token IDs 与参考引擎一致。
结果在 `artifacts/routing_smoke/summary.json`；原 P/D 服务的回归结果在
`artifacts/routing_pd_regression/summary.json`。这些功能检查不代表双卡性能结果。
