"""Show tinyserve 1.0's scheduler decisions during one real GPU request.

Run inside WSL with the project's environment:
    python trace_local.py
The printed timings/logs are for learning, not benchmarking.
"""

import argparse

from serve import Config, Engine, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="用一句话解释 KV cache 的作用。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=16)
    args = parser.parse_args()
    if args.token_budget < 4:
        parser.error("--token-budget must be at least 4")

    engine = Engine(Config(
        model=args.model,
        attention_backend="torch",
        use_cuda_graphs=False,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=args.token_budget,
        gpu_memory_utilization=0.35,
    ))
    initial_free_blocks = engine.kv.num_free
    if engine.tokenizer.chat_template:
        prompt_ids = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        print("No chat template: using the prompt as plain text.")
        prompt_ids = engine.tokenizer.encode(args.prompt)
    print(f"\nREQUEST: prompt_tokens={len(prompt_ids)}, budget={args.token_budget}")
    engine.add_request(
        prompt_ids,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        request_id="A",
    )

    original_schedule = engine.scheduler.schedule

    def traced_schedule():
        batch = original_schedule()
        for item in batch:
            phase = "prefill" if item.num_computed < item.req.prompt_len else "decode"
            print(
                f"  schedule {item.req.request_id}: {phase}, "
                f"KV_before={item.num_computed}, compute={item.num_new}, "
                f"pages={item.req.block_ids}, emit={item.completes_prompt}"
            )
        return batch

    engine.scheduler.schedule = traced_schedule
    step = 0
    final = None
    while engine.has_work():
        step += 1
        print(f"STEP {step}")
        outputs = engine.step()
        if not outputs:
            print("  KV updated; prompt is incomplete, no output token yet.")
        for output in outputs:
            print(
                f"  output token_id={output.token_ids[-1]}, "
                f"delta={output.delta!r}, finished={output.finished}"
            )
            final = output
    print(f"\nANSWER: {final.text if final else ''}")
    print(f"FREE KV BLOCKS: {engine.kv.num_free} / {initial_free_blocks}")


if __name__ == "__main__":
    main()
