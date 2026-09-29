"""ingest 模块单元测试。直接运行： python tests/test_ingest.py（无需 pytest）。"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg import (  # noqa: E402
    Node,
    NodeType,
    Ontology,
    Relation,
    RelationType,
    KnowledgeGraph,
    ingest_file,
    load_ontology_merged,
    merge_payload,
)


def base_onto() -> Ontology:
    return Ontology(
        nodes=[
            Node(id="transformer", name="Transformer", type=NodeType.MODEL,
                 domain="深度学习", difficulty=4, aliases=["变形器"]),
            Node(id="attention", name="注意力机制", type=NodeType.CONCEPT, domain="深度学习",
                 desc="按相关性动态加权聚合信息"),
            Node(id="positional_encoding", name="位置编码", type=NodeType.CONCEPT, domain="深度学习"),
            Node(id="linear_algebra", name="线性代数", type=NodeType.CONCEPT, domain="数学基础"),
        ],
        relations=[
            Relation(source="attention", target="transformer", type=RelationType.PREREQUISITE_OF),
            Relation(source="linear_algebra", target="attention", type=RelationType.PREREQUISITE_OF),
        ],
    )


def rel(s, t, rtype, conf=1.0):
    return {"source": s, "target": t, "type": rtype, "confidence": conf}


# ---------------- 消歧 ----------------


def test_resolve_by_id_and_merge_aliases():
    payload = {"nodes": [{"id": "attention", "name": "注意力机制", "type": "Concept",
                          "aliases": ["Attention"]}]}
    new_onto, rep = merge_payload(base_onto(), payload)
    assert rep.added_nodes == [], "同 id 不应新建"
    assert "attention" in rep.updated_nodes
    node = next(n for n in new_onto.nodes if n.id == "attention")
    assert "Attention" in node.aliases


def test_resolve_by_name_and_alias_in_relations():
    payload = {"relations": [
        rel("positional_encoding", "Transformer", "prerequisite_of"),  # 名称指称
        rel("变形器", "attention", "related_to"),  # 别名指称
    ]}
    _onto, rep = merge_payload(base_onto(), payload)
    assert rep.rejected == [], f"不应有拒绝: {rep.rejected}"
    assert "positional_encoding-[prerequisite_of]->transformer" in rep.added_relations
    assert "transformer-[related_to]->attention" in rep.added_relations


def test_fuzzy_match_unique_best():
    payload = {"relations": [rel("Transformers", "attention", "related_to")]}
    _onto, rep = merge_payload(base_onto(), payload)
    assert rep.fuzzy_resolved, "唯一最佳模糊命中应被采用"
    assert rep.rejected == []


def test_new_entity_chinese_name_gets_generated_id():
    payload = {"nodes": [{"name": "混合专家", "type": "Technique", "domain": "大模型工程"}]}
    new_onto, rep = merge_payload(base_onto(), payload)
    assert len(rep.added_nodes) == 1
    nid = rep.added_nodes[0]
    node = next(n for n in new_onto.nodes if n.id == nid)
    assert node.name == "混合专家" and nid != ""


def test_new_entity_id_collision_gets_suffix():
    payload = {"nodes": [
        {"id": "rope", "name": "旋转位置编码A", "type": "Technique"},
        {"id": "rope", "name": "旋转位置编码B", "type": "Technique"},
    ]}
    new_onto, rep = merge_payload(base_onto(), payload)
    assert sorted(rep.added_nodes) == ["rope", "rope_2"]
    names = {n.id: n.name for n in new_onto.nodes}
    assert names["rope"] != names["rope_2"]


# ---------------- 合并语义 ----------------


def test_duplicate_relation_keeps_higher_confidence():
    onto = base_onto()
    onto.relations.append(Relation(source="linear_algebra", target="transformer",
                                   type=RelationType.RELATED_TO, confidence=0.5))
    payload = {"relations": [rel("linear_algebra", "transformer", "related_to", conf=0.9)]}
    _new_onto, rep = merge_payload(onto, payload)
    assert rep.updated_relations == ["linear_algebra-[related_to]->transformer"]
    kept = next(r for r in _new_onto.relations
                if (r.source, r.target, r.type.value) == ("linear_algebra", "transformer", "related_to"))
    assert kept.confidence == 0.9


def test_lower_confidence_does_not_overwrite():
    payload = {"relations": [rel("attention", "transformer", "prerequisite_of", conf=0.5)]}
    _new_onto, rep = merge_payload(base_onto(), payload)
    assert rep.updated_relations == [] and rep.added_relations == []


def test_existing_desc_not_overwritten_but_empty_filled():
    payload = {"nodes": [
        {"id": "attention", "name": "注意力机制", "type": "Concept", "desc": "错误的新描述"},
        {"id": "positional_encoding", "name": "位置编码", "type": "Concept", "desc": "补充的描述"},
    ]}
    new_onto, _rep = merge_payload(base_onto(), payload)
    by_id = {n.id: n for n in new_onto.nodes}
    assert by_id["attention"].desc != "错误的新描述"
    assert by_id["positional_encoding"].desc == "补充的描述"


# ---------------- 拒绝规则 ----------------


def test_self_loop_rejected():
    payload = {"relations": [rel("attention", "attention", "related_to")]}
    _new_onto, rep = merge_payload(base_onto(), payload)
    assert len(rep.rejected) == 1 and "自环" in rep.rejected[0]["reason"]


def test_dangling_endpoint_rejected():
    payload = {"relations": [rel("不存在的概念xyz", "attention", "related_to")]}
    _new_onto, rep = merge_payload(base_onto(), payload)
    assert len(rep.rejected) == 1 and "无法消歧" in rep.rejected[0]["reason"]


def test_cycle_creating_prerequisite_rejected():
    # 已有链 linear_algebra -> attention -> transformer，补 transformer -> linear_algebra 会成环
    payload = {"relations": [rel("transformer", "linear_algebra", "prerequisite_of")]}
    new_onto, rep = merge_payload(base_onto(), payload)
    assert len(rep.rejected) == 1 and "环路" in rep.rejected[0]["reason"]
    from kg.store import KnowledgeGraph

    assert KnowledgeGraph.from_ontology(new_onto).check_dag() == []


def test_schema_violation_rejected_not_crash():
    payload = {
        "nodes": [{"id": "bad", "name": "坏节点", "type": "Concept", "difficulty": 9}],
        "relations": [rel("attention", "transformer", "未知关系类型")],
    }
    _new_onto, rep = merge_payload(base_onto(), payload)
    assert len(rep.rejected) == 2
    assert all("schema 校验失败" in r["reason"] for r in rep.rejected)


# ---------------- 文件级入库与补丁层 ----------------


def _write_seed(path: Path) -> None:
    onto = base_onto()
    payload = {
        "nodes": [n.model_dump(mode="json", exclude_none=True) for n in onto.nodes],
        "relations": [r.model_dump(mode="json", exclude_none=True) for r in onto.relations],
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


DEMO_PAYLOAD = {
    "nodes": [
        {"id": "rope", "name": "旋转位置编码", "type": "Technique", "domain": "大模型工程",
         "difficulty": 4, "aliases": ["RoPE"]},
        {"id": "attention", "name": "注意力机制", "type": "Concept", "aliases": ["Attention"]},
    ],
    "relations": [rel("rope", "Transformer", "applied_in", conf=0.9)],
}


def test_ingest_file_patch_layer_and_idempotency(tmp=None):
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        seed = data_dir / "seed_ontology.yaml"
        _write_seed(seed)
        seed_bytes = seed.read_bytes()
        payload = data_dir / "p.json"
        payload.write_text(json.dumps(DEMO_PAYLOAD, ensure_ascii=False), encoding="utf-8")

        rep = ingest_file(payload, data_dir)
        assert rep.added_nodes == ["rope"]
        assert "attention" in rep.updated_nodes
        assert seed.read_bytes() == seed_bytes, "seed 本体必须只读"
        assert (data_dir / "ontology_generated.yaml").exists()

        # 装载合并视图：新节点在，seed 节点的别名补丁也生效
        onto, _files = load_ontology_merged(seed, data_dir / "ontology_generated.yaml")
        ids = {n.id for n in onto.nodes}
        assert "rope" in ids
        attention = next(n for n in onto.nodes if n.id == "attention")
        assert "Attention" in attention.aliases

        # 幂等：同样内容再入一次，零新增且不触发落盘
        gen_mtime = (data_dir / "ontology_generated.yaml").stat().st_mtime_ns
        rep2 = ingest_file(payload, data_dir)
        assert rep2.added_nodes == [] and rep2.added_relations == []
        assert (data_dir / "ontology_generated.yaml").stat().st_mtime_ns == gen_mtime

        # dry-run：有新增但不落盘
        more = {"nodes": [{"id": "moe", "name": "混合专家", "type": "Technique"}]}
        p2 = data_dir / "p2.json"
        p2.write_text(json.dumps(more, ensure_ascii=False), encoding="utf-8")
        rep3 = ingest_file(p2, data_dir, dry_run=True)
        assert rep3.added_nodes == ["moe"]
        assert "moe" not in {n.id for n in load_ontology_merged(
            seed, data_dir / "ontology_generated.yaml")[0].nodes}


def test_provenance_survives_incremental_patch_and_sqlite():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        seed = data_dir / "seed_ontology.yaml"
        _write_seed(seed)
        payload_path = data_dir / "p.json"
        for chunk, quote, aliases in (("chunk-0000", "第一段", ["Attention"]),
                                      ("chunk-0001", "第二段", [])):
            ref = {"document": "notes.md", "chunk_id": chunk, "quote": quote,
                   "quote_verified": True}
            payload_path.write_text(json.dumps({
                "nodes": [{"id": "attention", "name": "注意力机制", "type": "Concept",
                           "aliases": aliases, "evidence_refs": [ref]}],
                "relations": [{"source": "attention", "target": "transformer",
                               "type": "prerequisite_of", "evidence_refs": [ref]}],
            }, ensure_ascii=False), encoding="utf-8")
            ingest_file(payload_path, data_dir)
        onto, _ = load_ontology_merged(seed, data_dir / "ontology_generated.yaml")
        node = next(n for n in onto.nodes if n.id == "attention")
        relation = next(r for r in onto.relations if r.source == "attention"
                        and r.target == "transformer")
        assert "Attention" in node.aliases, "后一次字段补丁不能抹掉先前别名"
        assert len(node.evidence_refs) == len(relation.evidence_refs) == 2
        kg = KnowledgeGraph.from_ontology(onto)
        db = data_dir / "kg.sqlite"
        kg.save_sqlite(db)
        reloaded = KnowledgeGraph.from_sqlite(db)
        assert len(reloaded.node("attention").evidence_refs) == 2
        edge = next(e for e in reloaded.edges_of("attention")
                    if e["target"] == "transformer" and e["type"] == "prerequisite_of")
        assert len(edge["evidence_refs"]) == 2


def test_embedding_alignment_merges_near_duplicate():
    """opt-in 嵌入对齐：名称带噪声的近重复应合并；路径不提供时行为不变。"""
    from kg.embeddings import HashEmbedder

    payload = {"nodes": [{"id": "attn2", "name": "注意力机制！", "type": "Concept"}]}
    # 不带 embedder：difflib 路径，正常合并
    _onto, rep_difflib = merge_payload(base_onto(), payload)
    assert rep_difflib.added_nodes == []
    # 带 embedder：嵌入对齐路径，近重复合并进已有节点（无新信息 -> 不计更新）
    emb = HashEmbedder(dim=128)
    new_onto, rep = merge_payload(base_onto(), payload, embedder=emb)
    assert rep.added_nodes == [] and rep.rejected == []
    assert "attn2" not in {n.id for n in new_onto.nodes}, "近重复应并入 attention 而非新建"


def test_embedding_alignment_ambiguity_creates_new():
    """两个候选都接近但都不够近（低于阈值）时：宁可新建，不错并。"""
    from kg.embeddings import HashEmbedder

    payload = {"nodes": [{"id": "attn_mgr", "name": "注意力机制管理平台", "type": "Concept"}]}
    emb = HashEmbedder(dim=128)
    new_onto, rep = merge_payload(base_onto(), payload, embedder=emb)
    assert rep.added_nodes == ["attn_mgr"], f"低于阈值应新建: {rep.summary()}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"[FAIL] {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
