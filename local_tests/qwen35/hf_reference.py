"""Generate deterministic Hugging Face reference outputs for Qwen3.5."""

from __future__ import annotations

import argparse
import platform
import time

from common import correctness_workloads, workload_dict, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/home/u2601112087/models/Qwen3.5-0.8B",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if args.device == "npu":
        import torch_npu  # noqa: F401

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    results = []
    for workload in correctness_workloads():
        messages = [{"role": "user", "content": workload.prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(
            args.device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        if args.device == "npu":
            torch.npu.synchronize()
        elapsed = time.perf_counter() - started
        prompt_tokens = inputs.input_ids.shape[-1]
        output_ids = generated[0, prompt_tokens:].cpu().tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        results.append(
            {
                **workload_dict(workload),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(output_ids),
                "output_token_ids": output_ids,
                "output_text": text,
                "elapsed_seconds": elapsed,
            }
        )
        print(f"{workload.name}: {text!r} ({elapsed:.3f}s)", flush=True)

    write_json(
        args.output,
        {
            "backend": "huggingface",
            "model_dir": args.model_dir,
            "device": args.device,
            "dtype": args.dtype,
            "platform": platform.platform(),
            "max_new_tokens": args.max_new_tokens,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
