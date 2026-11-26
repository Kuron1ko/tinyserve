from dataclasses import dataclass


@dataclass(frozen=True)
class PDConfig:
    model: str
    devices: tuple[int, int] = (0, 1)
    transport: str = "nccl"
    attention_backend: str = "torch"
    block_size: int = 256
    prefill_chunk_tokens: int = 512
    max_batched_tokens: int = 1024
    long_active_limit: int = 4
    max_num_seqs: int = 16
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.80
    headroom_tokens: int = 256
    efloor_blocks: int = 2
    pending_kv_bytes: int = 256 * 1024 * 1024
    resume_kv_bytes: int = 128 * 1024 * 1024
    expand_priority_ms: int = 100
    expand_timeout_ms: int = 5000
    request_timeout_s: float = 120
    startup_timeout_s: float = 300
    communication_timeout_s: float = 60
    max_queued_requests: int = 256

    def __post_init__(self):
        if self.block_size != 256:
            raise ValueError("This version uses 256-token KV pages.")
        if self.transport not in ("nccl", "gloo"):
            raise ValueError("transport must be nccl or gloo (CPU staging)")
        if len(self.devices) != 2 or min(self.devices) < 0:
            raise ValueError("Exactly two nonnegative device indices are required")
        if self.transport == "nccl" and self.devices[0] == self.devices[1]:
            raise ValueError("NCCL workers require distinct GPUs; use gloo for single-GPU functional testing")
        if self.attention_backend not in ("torch", "flash_attn"):
            raise ValueError("Unsupported attention backend")
        for name in ("prefill_chunk_tokens", "max_batched_tokens"):
            value = getattr(self, name)
            if value < self.block_size or value % self.block_size:
                raise ValueError(f"{name} must be a positive multiple of block_size")
        if self.prefill_chunk_tokens > self.max_batched_tokens:
            raise ValueError("prefill_chunk_tokens cannot exceed max_batched_tokens")
        if min(self.max_num_seqs, self.long_active_limit, self.max_queued_requests) < 1:
            raise ValueError("Queue/sequence limits must be positive")
        if self.max_model_len < self.block_size or min(self.headroom_tokens, self.efloor_blocks) < 0:
            raise ValueError("Invalid context or reservation limits")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must lie in (0, 1)")
        if not 0 <= self.resume_kv_bytes < self.pending_kv_bytes:
            raise ValueError("Require 0 <= resume_kv_bytes < pending_kv_bytes")
        if not 0 <= self.expand_priority_ms < self.expand_timeout_ms:
            raise ValueError("Expansion priority threshold must precede failure timeout")
        if min(self.request_timeout_s, self.startup_timeout_s, self.communication_timeout_s) <= 0:
            raise ValueError("Timeouts must be positive")
