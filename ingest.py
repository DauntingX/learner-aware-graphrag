#!/usr/bin/env python
"""增量入库 CLI：把一份抽取产物合并进图谱的机器补丁层。

用法：
    python ingest.py data/inbox/triples_demo.json            # 入库
    python ingest.py data/inbox/triples_demo.json --dry-run  # 只出报告，不落盘

入库后机器补丁层是 data/ontology_generated.yaml（seed 本体只读不被改动），
随后运行 python build_graph.py 即可让新知识点进入 SQLite 与前端可视化。

输入格式见 data/inbox/triples_demo.json 顶部的注释。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kg import GENERATED_NAME, ingest_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="知识图谱增量入库")
    ap.add_argument("payload", help="抽取产物文件（.json 或 .yaml）")
    ap.add_argument("--dry-run", action="store_true", help="只出报告，不写补丁层")
    ap.add_argument("--embed-align", action="store_true",
                    help="实体消歧升级为名称向量对齐（需要 llm_config 或内置哈希降级）")
    args = ap.parse_args()

    payload = Path(args.payload)
    if not payload.exists():
        print(f"[FAIL] 文件不存在：{payload}")
        return 1

    embedder = None
    if args.embed_align:
        from kg.embeddings import get_embedder
        from kg.llm import load_config

        try:
            cfg = load_config()
        except Exception:  # noqa: BLE001  # 无配置也可用哈希向量对齐
            cfg = None
        embedder = get_embedder(cfg)

    print(f"入库 {payload.name}{'（dry-run，不落盘）' if args.dry_run else ''}"
          f"{'（嵌入对齐）' if embedder else ''}")
    try:
        report = ingest_file(payload, ROOT / "data", dry_run=args.dry_run, embedder=embedder)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {exc}")
        return 1

    print(report.summary())
    if report.rejected:
        return 1
    if args.dry_run:
        print("\ndry-run 结束，未写入任何文件。")
    else:
        gen = ROOT / "data" / GENERATED_NAME
        print(f"\n[OK] 已写入 {gen.relative_to(ROOT)}  运行 python build_graph.py 生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
