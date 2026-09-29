#!/usr/bin/env python
"""LLM 自动抽取 CLI（P1 完整版）：语料 -> LLM 三元组 -> 入库。

模型与 API 完全自定义，任何 OpenAI 兼容接口都行。三种配置途径（优先级从高到低）：
    1. 命令行参数  --base-url / --api-key / --model
    2. 配置文件    llm_config.yaml（模板见 llm_config.example.yaml，已被 .gitignore 忽略）
    3. 环境变量    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

用法：
    python extract.py data/corpus/demo_notes.md --dry-run   # 只抽取预览，不落任何文件
    python extract.py data/corpus/demo_notes.md             # 抽取 -> 写 inbox -> 入库
    python extract.py notes/ --no-ingest                    # 只写 data/inbox/，人工过目后再入库
    python extract.py notes.md --model glm-4-plus --api-key xxx --base-url https://open.bigmodel.cn/api/paas/v4

默认行为：抽取产物写到 data/inbox/<语料名>_triples.json 并立即合并进图谱补丁层；
随后 python build_graph.py 即可让新知识点进入 SQLite 与可视化。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kg import GENERATED_NAME, ingest_file, load_ontology_merged  # noqa: E402
from kg.extract import extract_note, extract_text  # noqa: E402
from kg.llm import LLMClient, LLMConfigError, LLMError, load_config  # noqa: E402

STRUCTURED_EXTS = {".md", ".markdown", ".docx"}  # 有标题结构 -> 按小节分片
CORPUS_EXTS = {".md", ".markdown", ".txt", ".docx"}


def resolve_inputs(inputs: list[str]) -> list[Path]:
    """展开输入：目录则递归收集 md/txt，文件直接用。"""
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*")
                                if f.suffix.lower() in CORPUS_EXTS and not f.name.startswith(".")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[WARN] 跳过不存在的路径：{p}")
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 知识三元组抽取（模型与 API 可自定义）")
    ap.add_argument("inputs", nargs="+", help="语料文件或目录（.md/.txt）")
    ap.add_argument("--config", help="LLM 配置文件路径（默认项目根 llm_config.yaml）")
    ap.add_argument("--base-url", help="API 地址，如 https://open.bigmodel.cn/api/paas/v4")
    ap.add_argument("--api-key", help="API 密钥")
    ap.add_argument("--model", help="模型名，如 glm-4-flash / deepseek-chat / qwen-plus")
    ap.add_argument("--temperature", type=float, help="采样温度（默认 0.2）")
    ap.add_argument("--chunk-chars", type=int, help="语料分片大小（默认 4000 字符）")
    ap.add_argument("--limit", type=int, help="最多抽取多少个分片（控制花费）")
    ap.add_argument("--glean", type=int, metavar="N",
                    help="补抽轮数：首抽后追问遗漏 N 次（GraphRAG 多轮抽取思想，默认 0）")
    ap.add_argument("--out", help="产物文件名（默认 data/inbox/<语料名>_triples.json）")
    ap.add_argument("--dry-run", action="store_true", help="只抽取并打印预览，不写任何文件")
    ap.add_argument("--no-ingest", action="store_true", help="产物只写进 inbox，不自动入库")
    args = ap.parse_args()

    files = resolve_inputs(args.inputs)
    if not files:
        print("[FAIL] 没找到可读的语料文件（支持 .md/.markdown/.txt/.docx）")
        return 1

    overrides = {k: v for k, v in {
        "base_url": args.base_url, "api_key": args.api_key, "model": args.model,
        "temperature": args.temperature, "chunk_chars": args.chunk_chars,
        "gleans": args.glean,
    }.items() if v is not None}
    try:
        cfg = load_config(args.config, **overrides)
    except LLMConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"模型：{cfg.model}   API：{cfg.base_url}")
    try:
        client = LLMClient(cfg)
    except LLMConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1

    onto, _files = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml",
                                        ROOT / "data" / GENERATED_NAME)
    print(f"已有图谱：{len(onto.nodes)} 节点（作为消歧上下文注入提示词）\n")

    all_nodes: list[dict] = []
    all_rels: list[dict] = []
    any_ok = False
    for f in files:
        if f.suffix.lower() in STRUCTURED_EXTS:
            print(f"抽取 {f.name}（结构化笔记：按标题小节分片）...")
            payload, stats = extract_note(str(f), client, cfg, onto, max_chunks=args.limit)
        else:
            text = f.read_text(encoding="utf-8", errors="replace")
            print(f"抽取 {f.name}（{len(text)} 字符）...")
            payload, stats = extract_text(text, client, cfg, onto,
                                          max_chunks=args.limit, document=f.name)
        print(f"  分片 {stats['ok_chunks']}/{stats['chunks']} 成功 -> "
              f"节点 {stats['nodes']}，关系 {stats['relations']}")
        for msg in stats["failed_chunks"]:
            print(f"  [WARN] {msg}")
        if stats["ok_chunks"]:
            any_ok = True
        all_nodes.extend(payload["nodes"])
        all_rels.extend(payload["relations"])

    if not any_ok:
        print("\n[FAIL] 所有分片都抽取失败，没有可入库的产物")
        return 1

    payload_out = {"source": ", ".join(f.name for f in files),
                   "nodes": all_nodes, "relations": all_rels}

    if args.dry_run:
        print(f"\n[dry-run] 抽取到 {len(all_nodes)} 节点 / {len(all_rels)} 关系，预览：")
        for n in all_nodes[:8]:
            print(f"  + {n.get('id')}  {n.get('name')}  ({n.get('type')}, 难度{n.get('difficulty')})")
        if len(all_nodes) > 8:
            print(f"  ... 其余 {len(all_nodes) - 8} 个节点")
        for r in all_rels[:8]:
            print(f"  + {r.get('source')}-[{r.get('type')}]->{r.get('target')}  (置信度 {r.get('confidence')})")
        if len(all_rels) > 8:
            print(f"  ... 其余 {len(all_rels) - 8} 条关系")
        print("\ndry-run 结束，未写任何文件。去掉 --dry-run 即可入库。")
        return 0

    inbox = ROOT / "data" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else inbox / f"{files[0].stem}_triples.json"
    out.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 产物已写入 {out.relative_to(ROOT)}")

    if args.no_ingest:
        print("按 --no-ingest 未入库。人工过目后运行： "
              f"python ingest.py data/inbox/{out.name}")
        return 0

    print("开始入库...")
    try:
        report = ingest_file(out, ROOT / "data")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 入库失败：{exc}")
        return 1
    print(report.summary())
    if report.rejected:
        print("\n有被拒绝的条目，可编辑上面的产物文件后重新运行 python ingest.py")
        return 1
    print("\n[OK] 入库完成。运行 python build_graph.py 让新知识点进入 SQLite 与可视化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
