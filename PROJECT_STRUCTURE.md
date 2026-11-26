# tinyserve PD 项目结构与双卡部署

本项目基于上游 `serve.py` 扩展同机双卡、固定 1P1D 推理服务。P 和 D 各加载
一份完整模型，分别负责 prefill 与 decode；并非将一个模型的权重切到两张卡上。
当前支持 Linux/WSL。这里的“远端部署”指把整个项目放到**同一台远端双卡机器**，
在该机器的两个终端运行 `p_server.py`、`d_server.py`。

## 1. 项目结构

```text
tinyserve/
├── PROJECT_STRUCTURE.md     # 本文件：目录、职责、调用链、部署步骤
├── README.md                # 从创建虚拟环境到远端启动、请求验证的完整流程
├── requirements.txt         # 固定版本的直接依赖（包括 torch / datasets）
├── scripts/
│   ├── setup_env.sh         # 创建隔离 .venv、安装 CUDA wheel 和依赖
│   └── check_env.py         # 导入、GPU、BF16 运算和后端自检
├── docs/
│   ├── PD_DESIGN.md         # PD 策略与等资源基线评测说明
│   └── README_UPSTREAM.md   # 保留上游原说明
├── serve.py                 # 上游完整引擎：模型、attention、KV、采样、混合调度
├── p_server.py              # 独立 P 入口：HTTP 服务 + P GPU worker
├── d_server.py              # 独立 D 入口：连接 P、获取配置、启动 D GPU worker
├── pd_server.py             # 共用 HTTP API；也保留单命令启动 P+D 的入口
├── baseline_server.py       # 单 GPU 混合 P+D 基线；发布 KV 资源快照
├── baseline_router.py       # FIFO 路由：空闲 KV 页最多优先 / 轮转
├── bench_pd.py              # 合成长短混合负载，采集 TTFT / ITL / 失败率
├── trace_local.py           # 原有本机调用链追踪工具
├── pd/
│   ├── config.py            # PDConfig：模型、GPU、通信与策略参数
│   ├── mixed.py             # 混合基线计算线程、GPU 绑定、HTTP token 适配、KV 快照
│   ├── routing.py           # 多副本选择策略：最大空闲页与轮转平局处理
│   ├── policy.py            # 长短队列、chunk 调度、D 预留/扩容、背压策略
│   ├── worker.py            # PrefillWorker / DecodeWorker 请求状态机
│   ├── transport.py         # 整页 KV 打包/导入，NCCL 或 Gloo 张量收发
│   ├── ipc.py               # 独立启动时的 Unix socket 控制通道与队列桥接
│   ├── standalone.py        # 独立 P/D 的握手、配置同步、进程生命周期
│   ├── runtime.py           # 请求路由、流式 token 分发、trace、worker 管理
│   └── __init__.py
├── tests/
│   ├── test_pd_policy.py    # 不需要 GPU 的调度、预留、扩容策略测试
│   ├── test_pd_kv.py        # KV 物理页重映射测试
│   ├── test_pd_metrics.py   # 延迟百分位与失败率统计测试
│   ├── test_pd_routing.py   # 路由选择、FIFO、并发、流转发和失败行为
│   ├── test_pd_ipc.py       # 控制消息顺序、断连、重复启动、启动超时
│   ├── smoke_pd.py          # 原单入口运行时的真实模型正确性检查
│   ├── timeout_pd.py        # 扩容超时与资源回收检查
│   ├── smoke_launchers.py   # 两个独立入口的 HTTP/SSE、取消、退出、参考输出检查
│   └── smoke_routing.py     # 负载感知对照、取消回收、参考输出和评测脚本检查
└── artifacts/              # 本机测试产生的 trace、评测数据、日志（不提交 git）
```

`bench_pd.py` 当前生成合成输入，还没有 ShareGPT/LongBench 数据加载器。

## 2. 两个入口如何配合

```text
客户端
  │ HTTP / SSE
  ▼
p_server.py：HTTP + P supervisor
  │ 本地进程队列                 ▲ D 的 token / metrics / error
  ▼                             │
P GPU worker ── 控制消息 ── Unix socket IPC ── D supervisor：d_server.py
  │                                                  │ 本地进程队列
  │ NCCL：KV 张量；或 Gloo：CPU 暂存后传输              ▼
  └─────────────────────────────────────────────► D GPU worker
```

两个脚本各自监督一个 GPU 子进程。后台线程桥接控制消息，让计算线程继续使用本地
进程队列；KV 张量不经过 JSON 或 HTTP。D 不单独监听 HTTP 端口，客户端只访问 P。
建立 torch.distributed 进程组时还使用自动选择的本机 TCP rendezvous 端口。

启动握手由 P 提供整份 `PDConfig` 和 rendezvous 地址。D 无需重复填写模型路径、
chunk、headroom、efloor 等参数，避免配置漂移。两端需要运行相同代码版本。

## 3. 远端部署后启动

### 3.1 准备环境

1. 将整个 `tinyserve/` 目录放到双卡机器，例如 `/opt/tinyserve`。
2. 使用同一个 Linux 用户、同一个 Python 环境运行两个入口。
3. 按根目录 README 创建隔离虚拟环境，并准备引擎支持的模型权重目录。
   本机功能验证使用 Qwen2.5-1.5B；不保证任意 Hugging Face 架构都能直接运行。
4. 安装依赖并检查两张 GPU：

```bash
cd /opt/tinyserve
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_env.py --min-gpus 2
```

GPU 数量应至少为 2。默认 attention 后端为 `torch`，不要求安装 FlashAttention；
若选择 `--attention-backend flash_attn`，需另行安装匹配当前环境的 `flash-attn`。

### 3.2 终端 A：P 端

```bash
cd /opt/tinyserve
source .venv/bin/activate
python p_server.py \
  --model /data/models/Qwen2.5-1.5B \
  --devices 0,1 \
  --transport nccl \
  --host 0.0.0.0 \
  --port 8000
```

将模型路径替换为实际路径。`--devices 0,1` 的第一个数字属于 P，第二个属于 D。
P 打印 `Waiting for d_server.py` 后等待 D 连接；两侧引擎均就绪后才启动 HTTP 接口。

也可以预先设置 `export MODEL=/data/models/Qwen2.5-1.5B`，随后运行
`python p_server.py`。这种写法默认只监听 `127.0.0.1:8000`。

### 3.3 终端 B：D 端

```bash
cd /opt/tinyserve
source .venv/bin/activate
python d_server.py
```

D 会接收 P 的配置并使用 GPU 1。无需第三个路由服务或额外 `torchrun` 命令。
也允许先运行 D，再运行 P；默认等待连接时间为 300 秒。

两端必须使用相同的 `CUDA_VISIBLE_DEVICES`。例如若需要映射到物理 GPU 2、3，
在两个终端都设置 `export CUDA_VISIBLE_DEVICES=2,3`，P 仍使用 `--devices 0,1`。
不要分别将两端设置成 `CUDA_VISIBLE_DEVICES=0` 与 `CUDA_VISIBLE_DEVICES=1`。

默认控制端点是 `/tmp/tinyserve-pd-<UID>/control.sock`（遵循系统临时目录设置）。
目录自动创建为 0700。同一用户要运行第二套实例时，两端传入同一个新的目录，
例如 `--ipc-dir /tmp/tinyserve-pd-exp2`，并为 P 指定不同 HTTP 端口。
不要把 IPC 目录设到跨机共享存储；它用于本机 Unix socket。

### 3.4 验证服务

在远端机器本地执行：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is", "max_tokens":32, "temperature":0, "stream":true}'
```

健康检查返回 `{"status":"ok"}`。从其他机器访问时，将 `127.0.0.1` 换成远端地址，
并确保服务端口可达。还支持 `/v1/chat/completions`；聊天接口要求模型具有 chat template。

两张 4090 没有 NVLink。仍可尝试 NCCL，但实际是否走 GPU P2P 或主机中转需在目标
机器实测。如 NCCL 不可用，停止两端后，只在 P 启动命令中改为 `--transport gloo`；
D 自动跟随。这是显式的 CPU 暂存传输路径，不是自动回退。

## 4. 策略参数在哪里改

日常实验优先修改 P 的启动参数；默认值在 `pd/config.py`。D 继承 P 的全部参数。

| 参数 | 当前默认值 | 含义 |
| --- | ---: | --- |
| `--block-size` | 256 | KV 页大小，首版固定；扩容一次一页 |
| `--prefill-chunk-tokens` | 512 | chunk 大小，也是原始请求长短分界 |
| `--max-batched-tokens` | 1024 | P 一次 forward 的总 token 预算 |
| `--long-active-limit` | 4 | active 长请求容量 |
| `--max-num-seqs` | 16 | 并发执行/引擎容量配置 |
| `--max-model-len` | 8192 | 输入 + 输出最大 token 数 |
| `--headroom-tokens` | 256 | D 初次预留额外容量 |
| `--efloor-blocks` | 2 | 仅扩容可动用的保底页数 |
| `--pending-kv-bytes` | 268435456 | P 背压高水位，256 MiB |
| `--resume-kv-bytes` | 134217728 | 背压恢复水位，128 MiB |
| `--expand-priority-ms` | 100 | 扩容等待优先级阈值 |
| `--expand-timeout-ms` | 5000 | 扩容超时终止该请求 |
| `--request-timeout-s` | 120 | 请求总超时 |
| `--gpu-memory-utilization` | 0.8 | 每个引擎的显存利用率配置 |
| `--startup-timeout-s` | 300 | P 等待 D / worker 就绪的期限；D 上单独控制等待 P |
| `--communication-timeout-s` | 60 | 分布式通信超时 |

chunk 和 batch 预算都必须是 256 的倍数，chunk 不超过 batch 预算。
完整参数见 `python p_server.py --help`。若需修改调度算法，编辑 `pd/policy.py`；
状态机处理在 `pd/worker.py`；KV 传输实现则在 `pd/transport.py`。

## 5. 单次请求调用链

1. P 的 HTTP 层分词、检查长度，调用 `runtime.generate()`，建立请求流。
2. P worker 接收请求，查前缀缓存，放入短队列或长请求 waiting/active 队列。
3. 调度器在预算内交替选择长短工作，调用上游 ModelRunner 完成 chunk forward。
4. P 向 D 申请 `prompt + headroom` 的块容量。D 按 efloor 与等待队列策略决定预留。
5. 获得预留后，P 通信线程按 FIFO 发送已完成 chunk 对应的整页 KV，携带有效长度。
6. D 将收到的 KV 写入自己的物理页，检查顺序与覆盖范围；全部 prompt 到达后 commit。
7. D 发回 commit ACK，并向客户端输出首 token。P 在传输完成且收到 commit ACK 后
   释放该请求引用；缓存中可复用的页仍由原有 KV/LRU 管理。
8. D 持续 decode，需要扩容时申请一页；扩容等待超时终止请求并回收资源，首版不抢占。
9. D 的 token 经控制 IPC 回到 P HTTP 层，以 SSE 或完整 JSON 返回。取消同步到两端。

## 6. 停止、故障与测试

正常关闭时先在 P 终端按 Ctrl+C，P 会通知 D 停止并清理本机 socket；D 入口随后退出。
单独停止 D 会让 P 的健康检查返回 503，并使正在执行的请求报错。异常退出后应重启
两端，不会自动接入新的 D 或恢复未完成请求。强制杀死 P 留下的 socket，会在下一次
成功取得目录锁后清理；第二个 P 不能抢占仍在运行的实例。

两个入口与原 `pd_server.py` 共用 HTTP、调度器和 GPU worker，但独立入口额外经过
控制桥接。做性能对比时固定启动方式，避免将控制路径差异混入策略实验。

```bash
# 无需下载模型的策略/控制通信测试。
python -m unittest discover -s tests -v

# 独立入口端到端检查：测试脚本会自动启动并停止 P、D，不要同时手动启动。
PYTHONPATH=. python tests/smoke_launchers.py --model /data/models/Qwen2.5-1.5B --devices 0,1

# 只有一张卡时：同 GPU 双进程 + Gloo，仅检查功能，不代表双卡性能。
PYTHONPATH=. python tests/smoke_launchers.py --model /data/models/Qwen2.5-1.5B --devices 0,0 --d-first

# 手动启动 P/D 后进行合成混合负载测试。
python bench_pd.py --model /data/models/Qwen2.5-1.5B \
  --urls http://127.0.0.1:8000 --requests 1000 --rate 2 \
  --short-tokens 256 --long-tokens 4096 --short-fraction 0.8 \
  --output-tokens 256 --output artifacts/mixed_pd.jsonl
```

独立入口测试检查整页边界与多 chunk 输入、非流式/流式一致性、客户端取消后资源回收、
关闭 P 后 D 退出，并和相同后端的原始引擎比较 token IDs。
当前可访问机器只有一张 GPU；双卡 NCCL 及目标机器的 TTFT / P99 ITL 需要部署后实测。
