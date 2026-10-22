"""Shared Qwen3.5 assignment-one workloads and JSON helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workload:
    name: str
    prompt: str
    expected_substring: str


def correctness_workloads() -> list[Workload]:
    filler = "这是用于验证长上下文读取能力的无关背景信息。" * 80
    return [
        Workload("short_math", "用一句话回答：1加1等于几？", "2"),
        Workload("chinese_fact", "只回答城市名：中国的首都是哪里？", "北京"),
        Workload(
            "english_exact",
            "Return exactly the uppercase word BLUE and nothing else.",
            "BLUE",
        ),
        Workload(
            "long_context_retrieval",
            f"{filler}\n课程验证码是 ASCEND-910C-27。"
            f"\n{filler}\n只回答课程验证码。",
            "ASCEND-910C-27",
        ),
    ]


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def workload_dict(workload: Workload) -> dict[str, str]:
    return asdict(workload)
