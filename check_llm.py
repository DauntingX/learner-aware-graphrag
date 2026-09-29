#!/usr/bin/env python
"""LLM 配置自检：一条命令验证 chat 与 embeddings 两路都通。

用法： python check_llm.py
前置： llm_config.yaml 已配置，或环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 已设置
      （Windows 设置用户级环境变量后需重启终端）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kg.llm import LLMClient, LLMConfigError, LLMError, load_config  # noqa: E402


def main() -> int:
    try:
        cfg = load_config()
    except LLMConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"对话模型 : {cfg.model}  @ {cfg.base_url}")
    print(f"嵌入模型 : {cfg.embedding_model or '（未配置 -> 检索用本地哈希向量降级）'}")

    client = LLMClient(cfg)

    # 1) chat 通路
    try:
        reply = client.complete("只回答两个字：通了", system="测试调用。")
        print(f"[OK] chat 通路 -> {reply.strip()[:40]!r}")
    except LLMError as exc:
        print(f"[FAIL] chat 通路：{exc}")
        return 1

    # 2) embeddings 通路（可选）
    if cfg.embedding_model:
        try:
            vecs = client.embeddings(["自注意力机制"])
            print(f"[OK] embeddings 通路 -> 维度 {len(vecs[0])}")
        except LLMError as exc:
            print(f"[WARN] embeddings 不可用（检索将用哈希向量降级）：{str(exc)[:120]}")
    else:
        print("[INFO] 未配 embedding_model：GraphRAG 检索用哈希向量，全链路仍可跑；"
              "想要语义检索质量可在 llm_config.yaml 加 embedding_model")

    print("\n全部就绪。下一步：")
    print("  python extract.py data/corpus/demo_notes.md --dry-run   # 看真实抽取效果")
    print("  python graph_rag.py --compare --rebuild                 # 检索评测（语义向量）")
    print("  python graph_rag.py --eval-answers                      # 答案侧评测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
