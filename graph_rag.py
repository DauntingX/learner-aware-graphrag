#!/usr/bin/env python
"""GraphRAG CLI：学习者感知的检索增强检索。

与朴素 RAG 的区别：向量召回只找种子 -> 沿图边扩展前置知识/易混淆概念 ->
掌握度加权重排（薄弱子图优先）-> 组装成带画像标注的教学上下文。

用法：
    python graph_rag.py "GraphRAG 是什么"          # local search：节点级检索
    python graph_rag.py "大模型工程都有什么" --global  # global search：社区摘要检索
    python graph_rag.py "RLHF 怎么训练" --answer   # 检索 + LLM 生成（需配好 llm_config）
    python graph_rag.py --compare                  # 朴素 RAG vs GraphRAG 对比评测
    python graph_rag.py "..." --rebuild            # 重建向量索引

嵌入配置：llm_config.yaml 里配了 embedding_model 就用 API 语义向量；
没配则自动降级为本地哈希向量（确定性、可离线跑评测）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kg import GENERATED_NAME, LearnerModel, load_ontology_merged  # noqa: E402
from kg.embeddings import get_embedder  # noqa: E402
from kg.ingest import KnowledgeGraph  # noqa: E402
from kg.rag import (  # noqa: E402
    ANSWER_SYSTEM,
    DEFAULT_MAX_CONTEXT,
    DEFAULT_PREREQ_DEPTH,
    DEFAULT_TOP_K,
    compare_rag,
    evaluate_answers,
    extract_query_keywords,
    global_search,
    load_or_build_index,
    local_search,
)

DATA = ROOT / "data"
INDEX_PATH = DATA / "vector_index.json"


def load_everything(args):
    onto, _files = load_ontology_merged(DATA / "seed_ontology.yaml",
                                        DATA / GENERATED_NAME)
    kg = KnowledgeGraph.from_ontology(onto)
    learner_path = DATA / "learner_state.json"
    if not learner_path.exists():
        learner_path = DATA / "learner_state.example.json"
    learner = LearnerModel.load(learner_path)
    try:
        cfg = None
        from kg.llm import load_config
        cfg = load_config(getattr(args, "config", None))
    except Exception:  # noqa: BLE001  # 无 LLM 配置也能跑（哈希嵌入降级）
        cfg = None
    embedder = get_embedder(cfg)
    return kg, learner, embedder, cfg


def ensure_index(kg, learner, embedder, rebuild: bool) -> dict:
    index, built = load_or_build_index(kg, learner, embedder, INDEX_PATH,
                                      rebuild=rebuild)
    if built:
        print("构建向量索引（节点 + 社区摘要）...")
        print(f"  [OK] {len(index['nodes'].ids)} 节点 / "
              f"{len(index['communities'].ids)} 社区 -> {INDEX_PATH.name}")
    return index


def print_result(r) -> None:
    print(f"\n查询：{r.query}   模式：{r.mode}")
    if r.seeds:
        print("种子（掌握度加权重排后）：")
        for s in r.seeds:
            print(f"  {s['name']:<14} 相似度 {s['sim']:.3f}  掌握度 {s['mastery']:.2f}  "
                  f"加权分 {s['score']:.3f}")
    print("\n--- 检索上下文 ---")
    print(r.context_text or "（空）")
    if r.stats:
        print("--- 统计 ---")
        print(json.dumps(r.stats, ensure_ascii=False))


def run_compare(args) -> int:
    kg, learner, embedder, _cfg = load_everything(args)
    index = ensure_index(kg, learner, embedder, args.rebuild)
    eval_data = json.loads((DATA / "rag_eval.json").read_text(encoding="utf-8"))
    compare_kw = {"top_k": args.top_k, "prereq_depth": args.prereq_depth}
    if args.max_context is not None:
        compare_kw["max_context"] = args.max_context
    result = compare_rag(kg, learner, index, embedder, eval_data["queries"],
                         **compare_kw)
    s = result["summary"]
    print(f"\n{'=' * 66}\n  朴素 RAG vs GraphRAG（{s['queries']} 条查询，"
          f"嵌入器：{type(embedder).__name__}）\n{'=' * 66}")
    rows = [
        ("golden 命中率", s["golden_hit_naive"], s["golden_hit_graph"]),
        ("前置链覆盖率", s["prereq_coverage_naive"], s["prereq_coverage_graph"]),
        ("已掌握内容占比（越低越好）", s["mastered_ratio_naive"], s["mastered_ratio_graph"]),
        ("上下文平均规模", s["context_size_naive"], s["context_size_graph"]),
    ]
    print(f"  {'指标':<26}{'朴素 RAG':>10}{'GraphRAG':>10}")
    for name, a, b in rows:
        print(f"  {name:<28}{a:>8}{b:>10}")
    print("\n  per-query 明细（Y=golden 命中）：")
    for r in result["per_query"]:
        print(f"    {r['query'][:22]:<24} 命中 {'Y' if r['golden_hit_naive'] else 'N'}->{'Y' if r['golden_hit_graph'] else 'N'}"
              f"  前置覆盖 {r['prereq_cov_naive']:.2f}->{r['prereq_cov_graph']:.2f}"
              f"  已掌握占比 {r['mastered_naive']:.2f}->{r['mastered_graph']:.2f}"
              f"  规模 {r['size_naive']}->{r['size_graph']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GraphRAG 检索")
    ap.add_argument("query", nargs="?", help="查询文本")
    ap.add_argument("--global", dest="mode_global", action="store_true",
                    help="全局检索（社区摘要粒度）")
    ap.add_argument("--answer", action="store_true", help="检索后调 LLM 生成回答")
    ap.add_argument("--compare", action="store_true", help="朴素 RAG vs GraphRAG 对比评测")
    ap.add_argument("--config", help="LLM 配置文件路径")
    ap.add_argument("--rebuild", action="store_true", help="强制重建向量索引")
    ap.add_argument("--eval-answers", action="store_true",
                    help="答案侧评测：RAGAS 式 LLM-as-judge（faithfulness/relevancy），需要 LLM")
    ap.add_argument("--rerank", action="store_true",
                    help="扩展填充交给 LLM 按教学相关性重排（默认按掌握度薄弱优先）")
    ap.add_argument("--no-keywords", action="store_true",
                    help="关闭 LLM 查询关键词扩展（默认配好 LLM 时自动开启）")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--max-context", type=int, default=None,
                    help="图检索节点预算；对比评测默认与朴素检索同为 6，普通查询默认 12")
    ap.add_argument("--prereq-depth", type=int, default=DEFAULT_PREREQ_DEPTH)
    args = ap.parse_args()

    if args.compare:
        return run_compare(args)
    if not args.query:
        ap.error("需要一个查询文本，或使用 --compare")

    kg, learner, embedder, cfg = load_everything(args)
    index = ensure_index(kg, learner, embedder, args.rebuild)

    # LLM 可用时：查询关键词扩展（LightRAG）；--rerank 时填充交给 LLM 重排
    client = None
    extra_queries = None
    rerank_client = None
    if cfg is not None:
        from kg.llm import LLMClient

        client = LLMClient(cfg)
        if not args.no_keywords:
            extra_queries = extract_query_keywords(args.query, client)
            if extra_queries:
                print(f"查询关键词扩展：{'、'.join(extra_queries)}")
        if args.rerank:
            rerank_client = client

    if args.mode_global:
        result = global_search(args.query, kg, learner, index, embedder)
        print_result(result)
        return 0

    result = local_search(args.query, kg, learner, index, embedder,
                          top_k=args.top_k,
                          max_context=args.max_context if args.max_context is not None else DEFAULT_MAX_CONTEXT,
                          prereq_depth=args.prereq_depth,
                          extra_queries=extra_queries, rerank_client=rerank_client)
    print_result(result)

    if args.eval_answers:
        if client is None:
            print("\n[FAIL] --eval-answers 需要 LLM（llm_config.yaml 或环境变量）")
            return 1
        eval_data = json.loads((DATA / "rag_eval.json").read_text(encoding="utf-8"))
        result_eval = evaluate_answers(kg, learner, index, embedder, client,
                                       eval_data["queries"])
        s = result_eval["summary"]
        print(f"\n{'=' * 60}\n  答案侧评测（RAGAS 式，{s['queries']} 条查询）\n{'=' * 60}")
        print(f"  {'指标':<24}{'朴素 RAG':>10}{'GraphRAG':>10}")
        for key, label in (("faithfulness", "忠实度"), ("relevancy", "切题度"),
                           ("golden_covered", "golden 覆盖率")):
            print(f"  {label:<26}{s[f'{key}_naive']:>10}{s[f'{key}_graph']:>10}")
        return 0

    if args.answer:
        if client is None:
            print("\n[WARN] 未配置 LLM（llm_config.yaml），跳过生成。检索上下文已在上方。")
            return 0
        prompt = f"图谱上下文：\n{result.context_text}\n\n用户问题：{args.query}"
        print("\n--- 生成回答 ---")
        print(client.complete(prompt, system=ANSWER_SYSTEM))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
