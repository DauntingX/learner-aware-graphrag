"""GraphRAG 检索引擎测试。直接运行： python tests/test_rag.py（无需 pytest、不打真实 API）。

覆盖：哈希嵌入确定性、向量索引持久化回环、local search 种子命中与图扩展、
掌握度加权填充（合成图精确控制）、global search、对比评测结构。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg import (LearnerModel, LearnerState, Node, NodeType, Ontology, Relation,
                RelationType)  # noqa: E402
from kg.schema import SourceRef  # noqa: E402
from kg import KnowledgeGraph, load_ontology_merged  # noqa: E402
from kg.embeddings import APIEmbedder, HashEmbedder, VectorIndex, embedder_identity  # noqa: E402
from kg.llm import LLMClient, LLMConfig, load_config  # noqa: E402
from kg.rag import (  # noqa: E402
    build_index,
    compare_rag,
    evaluate_answers,
    extract_query_keywords,
    global_search,
    index_is_current,
    load_index,
    load_or_build_index,
    local_search,
    node_index_text,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt, system=""):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeLLM 响应用完了")
        return self.responses.pop(0)


def real_graph():
    onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml",
                                   ROOT / "data" / "ontology_generated.yaml")
    kg = KnowledgeGraph.from_ontology(onto)
    learner = LearnerModel.load(ROOT / "data" / "learner_state.example.json")
    return kg, learner


# ---------------- 嵌入层 ----------------


def test_hash_embedder_deterministic_and_normalized():
    emb = HashEmbedder(dim=256)
    v1 = emb.embed_one("注意力机制是 Transformer 的核心")
    v2 = emb.embed_one("注意力机制是 Transformer 的核心")
    v3 = emb.embed_one("完全无关的文本内容")
    assert (v1 == v2).all(), "同文本嵌入必须确定性一致"
    assert abs(float((v1 @ v1)) - 1.0) < 1e-5, "L2 归一化后自相似应为 1"
    assert abs(float(v1 @ v3)) < 0.5, "无关文本相似度应显著低于自相似"
    batch = emb.embed_texts(["a", "b"])
    assert batch.shape == (2, 256)


def test_vector_index_roundtrip(tmp=None):
    emb = HashEmbedder(dim=128)
    idx = VectorIndex.build("hash", "hash", [
        ("a", "注意力机制 attention", {"name": "注意力"}),
        ("b", "梯度下降优化算法", {"name": "梯度下降"}),
    ], emb)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "vi.json"
        idx.save(p)
        loaded = VectorIndex.load(p)
    r1 = idx.search("注意力", emb, k=1)
    r2 = loaded.search("注意力", emb, k=1)
    # save 时向量保留 6 位小数，相似度允许 1e-5 级浮点差
    assert [i for i, _ in r1] == [i for i, _ in r2] == ["a"]
    assert abs(r1[0][1] - r2[0][1]) < 1e-5


def test_node_index_text_weights_name():
    onto = Ontology(nodes=[Node(id="x", name="循环神经网络", type=NodeType.MODEL,
                                domain="深度学习", desc="序列模型")], relations=[])
    text = node_index_text(onto.nodes[0])
    assert text.count("循环神经网络") == 3, "名称应重复 3 次做字段加权"


# ---------------- local search ----------------


def test_local_search_hits_seed_and_expands_prereqs():
    kg, learner = real_graph()
    emb = HashEmbedder()
    index = build_index(kg, learner, emb)
    r = local_search("RLHF 的训练流程是怎样的", kg, learner, index, emb)
    seed_ids = {s["id"] for s in r.seeds}
    ctx_ids = [n["id"] for n in r.context_nodes]
    assert "rlhf" in seed_ids, f"rlhf 应进种子，实际 {seed_ids}"
    # 图扩展：rlhf 的前置应进上下文（朴素 RAG 的结构性盲区）
    prereq_roles = [n for n in r.context_nodes if n["role"] == "prereq"]
    assert prereq_roles, "应扩展出前置知识"
    prereq_ids = {n["id"] for n in prereq_roles}
    assert {"sft", "reward_model"} & prereq_ids, f"RLHF 的直接前置应被扩展，实际 {prereq_ids}"
    # 上下文逐节点带掌握度标注
    assert all("mastery" in n and "band" in n for n in r.context_nodes)
    assert r.context_text.count("【") >= 2, "上下文应按角色分节"


def test_mastery_weighting_weak_first_fill():
    """合成图精确验证：同一种子的两个相似邻居，薄弱者先进上下文。"""
    nodes = [
        Node(id="hub", name="核心概念", type=NodeType.CONCEPT, domain="测试域", desc="核心",
             evidence_refs=[SourceRef(document="demo.md", chunk_id="demo.md#0",
                                      quote="核心概念", quote_verified=True)]),
        Node(id="strong", name="已掌握邻居", type=NodeType.CONCEPT, domain="测试域", desc="相关内容"),
        Node(id="weak", name="薄弱邻居", type=NodeType.CONCEPT, domain="测试域", desc="相关内容"),
    ]
    rels = [Relation(source="hub", target="strong", type=RelationType.RELATED_TO),
            Relation(source="hub", target="weak", type=RelationType.RELATED_TO)]
    kg = KnowledgeGraph.from_ontology(Ontology(nodes=nodes, relations=rels))
    learner = LearnerModel(states={})
    learner.states["strong"] = LearnerState(concept_id="strong", mastery=0.95)
    learner.states["weak"] = LearnerState(concept_id="weak", mastery=0.05)
    emb = HashEmbedder()
    index = build_index(kg, learner, emb)
    r = local_search("核心概念", kg, learner, index, emb, top_k=1, max_context=3)
    order = [n["id"] for n in r.context_nodes]
    assert order[0] == "hub"
    assert r.context_nodes[0]["evidence_refs"][0]["quote_verified"] is True
    assert order[1] == "weak", f"薄弱邻居应先进上下文（掌握度加权填充），实际 {order}"
    assert order[2] == "strong"


def test_local_search_empty_query_is_safe():
    kg, learner = real_graph()
    emb = HashEmbedder()
    index = build_index(kg, learner, emb)
    r = local_search("zzzqqqxyz 完全无关词", kg, learner, index, emb)
    assert r.context_nodes == [] or r.context_text != ""


# ---------------- global search ----------------


def test_global_search_returns_communities():
    kg, learner = real_graph()
    emb = HashEmbedder()
    index = build_index(kg, learner, emb)
    assert len(index["communities"].ids) >= 3, "应有多个社区"
    r = global_search("评估方法领域整体都有哪些东西", kg, learner, index, emb)
    assert r.stats.get("communities"), "应命中社区"
    assert r.context_text, "应有社区摘要文本"
    assert "社区 #" in r.context_text


def test_global_search_survives_index_roundtrip():
    kg, learner = real_graph()
    emb = HashEmbedder()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "index.json"
        built = build_index(kg, learner, emb, out_path=path)
        loaded = load_index(path)
        assert loaded["community_meta"] == built["community_meta"]
        assert index_is_current(loaded, kg, learner, emb)
        result = global_search("评估方法领域整体都有哪些东西", kg, learner, loaded, emb)
        assert result.context_text and result.context_nodes
        assert "社区 #" in result.context_text


def test_index_rebuilds_for_graph_mastery_model_and_partial_files():
    kg, learner = real_graph()
    emb = HashEmbedder()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "index.json"
        first, built = load_or_build_index(kg, learner, emb, path)
        assert built
        second, built = load_or_build_index(kg, learner, emb, path)
        assert not built and second["nodes"].cache_signature == first["nodes"].cache_signature

        kg.g.nodes["rlhf"]["desc"] += "（已更新）"
        updated, built = load_or_build_index(kg, learner, emb, path)
        assert built and updated["nodes"].cache_signature != first["nodes"].cache_signature

        learner.states["rlhf"] = LearnerState(concept_id="rlhf", mastery=0.42)
        updated_again, built = load_or_build_index(kg, learner, emb, path)
        assert built and updated_again["nodes"].cache_signature != updated["nodes"].cache_signature

        other_dim = HashEmbedder(dim=256)
        resized, built = load_or_build_index(kg, learner, other_dim, path)
        assert built and resized["nodes"].vectors.shape[1] == 256
        assert not index_is_current(resized, kg, learner, emb)

        Path(str(path) + ".communities.json").unlink()
        repaired, built = load_or_build_index(kg, learner, other_dim, path)
        assert built and repaired["community_meta"]


def test_api_embedding_identity_tracks_model_and_endpoint_without_secret():
    first = LLMConfig(base_url="https://api.example.com/v1", api_key="secret",
                      model="chat", embedding_model="embedding-a")
    other_model = LLMConfig(base_url=first.base_url, api_key="secret", model="chat",
                            embedding_model="embedding-b")
    other_endpoint = LLMConfig(base_url="https://other.example.com/v1", api_key="secret",
                               model="chat", embedding_model="embedding-a")
    keys = [embedder_identity(APIEmbedder(LLMClient(cfg)))
            for cfg in (first, other_model, other_endpoint)]
    assert len(set(keys)) == 3
    assert all("secret" not in str(key) and "example.com" not in str(key) for key in keys)


# ---------------- 对比评测 ----------------


def test_compare_rag_structure_and_prereq_win():
    kg, learner = real_graph()
    emb = HashEmbedder()
    index = build_index(kg, learner, emb)
    eval_data = json.loads((ROOT / "data" / "rag_eval.json").read_text(encoding="utf-8"))
    result = compare_rag(kg, learner, index, emb, eval_data["queries"])
    s = result["summary"]
    assert 0.0 <= s["prereq_coverage_naive"] <= s["prereq_coverage_graph"] <= 1.0
    assert s["prereq_coverage_graph"] > s["prereq_coverage_naive"] + 0.05, \
        f"图扩展应显著提升前置链覆盖: {s}"
    assert all(r["size_graph"] <= 6 for r in result["per_query"])
    assert set(result["per_query"][0]) >= {"query", "golden_hit_naive",
                                           "prereq_cov_graph", "mastered_graph"}


# ---------------- API 嵌入（协议解析，不发网络请求） ----------------


def test_llm_embeddings_response_parsing(tmp=None):
    config = LLMConfig(base_url="https://api.example.com/v1", api_key="k",
                       model="m", embedding_model="emb-1")
    client = LLMClient(config)
    captured = {}

    def fake_post(url, body):
        captured["url"] = url
        captured["body"] = body
        return {"data": [
            {"index": 1, "embedding": [0.0, 1.0, 0.0]},
            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
        ]}

    client._post = fake_post  # monkeypatch：不发真实请求
    vecs = client.embeddings(["甲", "乙"])
    assert captured["url"].endswith("/embeddings")
    assert captured["body"]["model"] == "emb-1"
    assert len(vecs) == 2
    assert abs(vecs[0][0] - 1.0) < 1e-6, "应按 index 排序并对齐输入顺序"
    api_emb = APIEmbedder(client)
    arr = api_emb.embed_texts(["甲", "乙"])
    assert arr.shape == (2, 3) and abs(float((arr[0] @ arr[0]))) - 1.0 < 1e-5


def test_config_embedding_fields():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "c.yaml"
        f.write_text(
            "base_url: https://api.example.com/v1\n"
            "api_key: k\nmodel: m\nembedding_model: emb-1\n", encoding="utf-8")
        cfg = load_config(f)
        assert cfg.embedding_model == "emb-1"
        assert cfg.embedding_base_url == "https://api.example.com/v1", \
            "embedding_base_url 缺省应沿用 base_url"
        # 不配 embedding_model -> 哈希降级
        f.write_text("base_url: https://api.example.com/v1\napi_key: k\nmodel: m\n",
                     encoding="utf-8")
        cfg2 = load_config(f)
        assert cfg2.embedding_model == ""


# ---------------- 查询扩展 / rerank / 答案评测 ----------------


def test_multi_query_feedback_preserves_intent():
    """伪反馈应把症状相关节点带进上下文，但不得挤掉原查询的 top-1（意图保底）。"""
    kg, learner = real_graph()
    emb = HashEmbedder()
    index = build_index(kg, learner, emb)
    q = "模型训练时梯度消失怎么办"
    base_first = index["nodes"].search(q, emb, k=1)[0][0]
    r = local_search(q, kg, learner, index, emb)  # 默认开启伪反馈
    assert r.seeds[0]["id"] == base_first, "原查询 top-1 必须保底为种子第 1"
    ctx_ids = {n["id"] for n in r.context_nodes}
    assert {"lstm", "backpropagation"} & ctx_ids, f"症状相关节点应被带进上下文: {ctx_ids}"


def test_extract_query_keywords_fake():
    good = json.dumps({"keywords": ["KV 缓存", "推理优化"]}, ensure_ascii=False)
    assert extract_query_keywords("推理时 KV 缓存怎么优化", FakeLLM([good])) == ["KV 缓存", "推理优化"]
    assert extract_query_keywords("q", FakeLLM(["完全不是 JSON"])) == [], "解析失败应退化为空"


def test_rerank_reorders_fill_and_falls_back():
    nodes = [
        Node(id="hub", name="核心概念", type=NodeType.CONCEPT, domain="测试域", desc="核心"),
        Node(id="strong", name="已掌握邻居", type=NodeType.CONCEPT, domain="测试域", desc="相关内容"),
        Node(id="weak", name="薄弱邻居", type=NodeType.CONCEPT, domain="测试域", desc="相关内容"),
    ]
    rels = [Relation(source="hub", target="strong", type=RelationType.RELATED_TO),
            Relation(source="hub", target="weak", type=RelationType.RELATED_TO)]
    kg = KnowledgeGraph.from_ontology(Ontology(nodes=nodes, relations=rels))
    learner = LearnerModel(states={})
    learner.states["strong"] = LearnerState(concept_id="strong", mastery=0.95)
    learner.states["weak"] = LearnerState(concept_id="weak", mastery=0.05)
    emb = HashEmbedder()
    index = build_index(kg, learner, emb)

    # 默认：薄弱优先
    r0 = local_search("核心概念", kg, learner, index, emb, top_k=1, max_context=3)
    assert [n["id"] for n in r0.context_nodes] == ["hub", "weak", "strong"]
    # rerank：LLM 要求已掌握的排前 -> 顺序翻转
    fake = FakeLLM(['["strong", "weak"]'])
    r1 = local_search("核心概念", kg, learner, index, emb, top_k=1, max_context=3,
                      rerank_client=fake)
    assert [n["id"] for n in r1.context_nodes] == ["hub", "strong", "weak"]
    # rerank 解析失败：退回薄弱优先，不报错
    fake2 = FakeLLM(["这不是数组"])
    r2 = local_search("核心概念", kg, learner, index, emb, top_k=1, max_context=3,
                      rerank_client=fake2)
    assert [n["id"] for n in r2.context_nodes] == ["hub", "weak", "strong"]


def test_evaluate_answers_fake():
    kg, learner = real_graph()
    emb = HashEmbedder()
    index = build_index(kg, learner, emb)
    ans = "自注意力是 Transformer 的核心机制。"
    judge = json.dumps({"faithfulness": 0.9, "relevancy": 0.8})
    fake = FakeLLM([ans, judge, ans, judge])  # naive: 答案+判卷；graph: 答案+判卷
    eval_q = [{"query": "自注意力是什么", "golden": ["self_attention"]}]
    res = evaluate_answers(kg, learner, index, emb, fake, eval_q)
    s = res["summary"]
    assert s["queries"] == 1 and s["faithfulness_naive"] == 0.9 and s["relevancy_graph"] == 0.8
    assert s["golden_covered_naive"] == 1.0, "回答包含 golden 概念名应计覆盖"
    assert len(fake.prompts) == 4, "每个方法两次调用：生成 + 判卷"


def main() -> int:
    import inspect
    import traceback

    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
