#!/usr/bin/env python
"""前置关系补全建议（P2）：LLM 找图谱缺口 -> 产出待审核产物 -> 人工入库。

用法：
    python suggest.py --dry-run        # 只打印建议，不写文件
    python suggest.py                  # 写 data/inbox/suggested_prereqs.json
    python ingest.py data/inbox/suggested_prereqs.json   # 人工过目后入库

为什么不让它直接入库：自动建议的边可能想当然（比如硬造「线性代数的前置」），
产物流经 inbox 的设计让「机器建议」和「人工决策」分离 —— 建议可以错，图谱不能错。
入库时 ingest 的成环拒绝仍是最后防线。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kg import GENERATED_NAME, load_ontology_merged  # noqa: E402
from kg.llm import LLMClient, LLMConfigError, LLMError, load_config  # noqa: E402
from kg.suggest import suggest_prerequisites  # noqa: E402

DATA = ROOT / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 前置关系补全建议")
    ap.add_argument("--config", help="LLM 配置文件路径")
    ap.add_argument("--limit", type=int, default=12, help="检查多少个无前置候选节点")
    ap.add_argument("--max-edges", type=int, default=10, help="最多建议多少条边")
    ap.add_argument("--dry-run", action="store_true", help="只打印建议，不写文件")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
        client = LLMClient(cfg)
    except LLMConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1

    onto, _files = load_ontology_merged(DATA / "seed_ontology.yaml", DATA / GENERATED_NAME)
    print(f"图谱 {len(onto.nodes)} 节点，检查 {min(args.limit, len(onto.nodes))} 个无前置候选...\n")

    try:
        payload, stats = suggest_prerequisites(onto, client, limit=args.limit,
                                               max_edges=args.max_edges)
    except (LLMError, Exception) as exc:  # noqa: BLE001
        print(f"[FAIL] 建议失败：{exc}")
        return 1

    print(f"候选 {stats['candidates']} 个 -> 建议 {stats['suggested']} 条前置边：\n")
    for e in payload["relations"]:
        print(f"  + {e.get('source')} -[prerequisite_of]-> {e.get('target')}"
              f"  置信度 {e.get('confidence')}")
        print(f"      理由：{e.get('note', '')}")

    if args.dry_run:
        print("\ndry-run 结束，未写文件。")
        return 0
    if not payload["relations"]:
        print("\nLLM 认为没有缺失的前置关系，无需入库。")
        return 0

    out = DATA / "inbox" / "suggested_prereqs.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 建议已写入 {out.relative_to(ROOT)}")
    print("请人工过目后执行： python ingest.py data/inbox/suggested_prereqs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
