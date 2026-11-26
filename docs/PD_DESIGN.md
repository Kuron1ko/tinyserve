## 本项目新增：同机双卡 PD 分离

环境创建与依赖安装统一按 [根目录 README](../README.md) 执行；本页命令均在项目根目录运行。

上游 `serve.py` 保留原样。新增实现位于 `pd/`。独立启动使用 `p_server.py` +
`d_server.py`；原 `pd_server.py` 仍可一次启动两端。
项目结构、调用链和远端双卡部署步骤见 [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)。
该实现固定 1P1D，每个 worker 加载完整模型。控制消息走进程队列 IPC；
整页 KV 张量走 NCCL，或显式选择 Gloo + CPU 暂存。使用 NCCL 不代表一定走
NVLink/GPU P2P，实际路径取决于硬件、驱动与 NCCL；4090 无 NVLink。
独立启动时，两侧本地进程队列通过 Unix socket 控制通道连接，KV 张量仍走上述数据通道。

### 已确认的策略

* `block_size=256`，扩容一次一页，即 256 token。固定 chunk 和 batch token
  预算是两个参数，均为 256 的倍数；最终不足一页的真实 token 不补入 prompt。
* P 在一个 packed batch 内按短、长轮流选择工作，偏好延续到下一轮；一个请求
  在一次 forward 中最多贡献一个 chunk。某队列无法使用剩余预算时交给另一队列，
  预算不足时缩到整页。短队列 FCFS；长请求 active 最多 4 个，waiting FIFO，
  active 按 chunk 轮转，剩余不超过 chunk 时移到短队列尾部。
* P 保留上游前缀缓存和空闲页 LRU；缓存命中的 KV 同样发送给 D。D 无前缀复用。
* D 每请求只成功预留一次 `prompt + headroom`，headroom 不超出最大上下文。
  在扣除之前判断 `free_blocks > requested_blocks + efloor_blocks`。
  扩容可使用 efloor。先服务扩容，再按 FIFO 服务新预留；当前扩容大小相同，
  扩容顺序为到达顺序。等待优先级阈值与失败超时分开配置。
* 扩容超时后终止请求并回收资源，首版不抢占。恢复/抢占策略可扩展
  `DecodeMemoryPolicy` 的事件处理。整个请求还有总期限；不可能满足的预留立即报错。
* 只有 KV 已就绪且已获 D 预留的 chunk 才进入发送 FIFO；未获预留的内容单独等待。
  P 可继续计算尚未获预留的请求，直到背压阈值。通信线程按实际入队次序串行发送，
  使用独立 CUDA stream，可与后续 forward 重叠；实际重叠效果需要 profile。
* P 待交接数据超过高水位时，停止引擎准入与 forward，但继续 IPC、传输和清理；
  降到恢复水位后继续。水位按尚未 ACK 的唯一源 KV 页计数，已 ACK 但后续 prefill
  仍需读取的历史页继续保留，不属于待交接队列。
* chunk ACK 只结束该 chunk 的传输占用。P 必须完成整个 prompt 且收到最终
  commit ACK 后才能释放请求引用。D 在所有 chunk 导入完成后 commit，随后发送
  首 token；其第一轮 decode 只计算这个 token 的 KV，不重新 prefill。
* 取消在 commit 前后都传播到两端；已宣布的张量接收必须排空，避免 NCCL 消息错位。
  请求失败回收资源；通信器/worker 故障使当前服务报错，需要重启服务，不自动重试。

这些数值是可调实验初值，不代表最优配置：chunk=512、batch=1024、headroom=256、
efloor=2 页、背压高/低水位=256/128 MiB、扩容优先级阈值=100 ms、扩容失败=5 s。
首版固定 BF16、关闭 CUDA Graph；默认使用 torch 参考 attention，可显式选择
已安装且兼容的 flash_attn。基线必须使用相同后端、精度、页大小和上下文配置。

### 启动与正确性检查

在 Linux/WSL 的项目环境中运行。沿用上游依赖；评测额外使用 `httpx`。
先将 `MODEL` 设置为同一份本地模型目录或 Hugging Face 模型标识。

```bash
export MODEL=/path/to/model
python -m pip install httpx
python -m unittest discover -s tests -v

# 两张 GPU；先做正确性检查，再进行评测。
PYTHONPATH=. python tests/smoke_pd.py --model "$MODEL" --devices 0,1
PYTHONPATH=. python tests/timeout_pd.py --model "$MODEL" --devices 0,1
python pd_server.py --model "$MODEL" --devices 0,1 --transport nccl

# 如目标平台 NCCL 无法使用，可显式改用主机暂存路径。
python pd_server.py --model "$MODEL" --devices 0,1 --transport gloo
```

本次开发环境只有一张 4090 Laptop。以下模式用同一 GPU 的两个进程验证请求
交接、页重映射和状态机，**不能用于宣称双卡加速或验证 NCCL**：

```bash
PYTHONPATH=. python tests/smoke_pd.py --model "$MODEL" --devices 0,0
python pd_server.py --model "$MODEL" --devices 0,0 --transport gloo \
  --max-model-len 768 --max-num-seqs 4 --max-batched-tokens 512 \
  --prefill-chunk-tokens 256 --gpu-memory-utilization 0.4
```

接口包括 `/health`、`/metrics`、`/v1/models`、`/v1/completions` 和
`/v1/chat/completions`，支持 SSE。流式响应每个生成 token 对应一个事件，附带
`x_token_id`；非流式请求可设置 `return_token_ids=true` 以检查精确 token 序列。
聊天模型可通过 `chat_template_kwargs` 禁用 thinking。请求超时/扩容失败会明确返回错误，
不会静默重算或抢占。

`--trace path.jsonl` 保存阶段、chunk、页数和 token 事件，用于调试；纯性能测试
可关闭 trace，避免额外记录开销。`--help` 列出所有策略参数。

接收暂存队列最多保留两个已完成的 GPU payload，单轮最多处理两个接收完成事件，
防止通信持续快于解码时无限积累 GPU 暂存数据。此上限不改变 chunk 的 FIFO 次序。

### 双卡等资源评测

PD 和基线分开运行。基线是在两张卡上分别启动完整 P+D 引擎，通过两个 URL
轮流分发请求；不要把双卡 PD 与单卡基线直接比较。下面命令在不同终端启动：

```bash
# 基线：两张卡，各一个混合实例。
python baseline_server.py --model "$MODEL" --device 0 --port 8001
python baseline_server.py --model "$MODEL" --device 1 --port 8002
python bench_pd.py --model "$MODEL" \
  --urls http://127.0.0.1:8001,http://127.0.0.1:8002 \
  --requests 100 --rate 2 --output artifacts/mixed.jsonl

# 停掉两个基线实例，再启动双卡 PD。
python pd_server.py --model "$MODEL" --devices 0,1 --transport nccl
python bench_pd.py --model "$MODEL" --urls http://127.0.0.1:8000 \
  --requests 100 --rate 2 --output artifacts/pd.jsonl
```

评测采用固定随机种子的泊松到达序列，默认 80% 128-token 短输入和 20%
2048-token 长输入，固定生成 128 token。默认加入不同请求前缀避免无意的全量缓存命中；
`--shared-prefix-tokens` 可控制共享前缀长度。每个请求保存发送时间、每个 token 的
客户端接收时间和错误。汇总输出 TTFT P50/P95/P99、所有 token 间隔的 P99、
首二 token 间隔 P99、失败率和客户端派发延迟；失败不会被伪装成成功。
短 smoke run 的统计仅用于验证工具，P99 结论需要足够样本和多次运行。

依次改变请求速率、输入长度组合、chunk、headroom 与 efloor，保持比较的其他条件
一致。测试脚本不能证明性能优势；应报告在哪些工作负载上改善、在哪些情况下变差。
