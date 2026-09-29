"""P2 前置关系补全测试。直接运行： python tests/test_suggest.py（无需 pytest、不打真实 API）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg import load_ontology_merged, merge_payload  # noqa: E402
from kg.extract import ExtractError  # noqa: E402
from kg.paths import prerequisite_graph  # noqa: E402
from kg.store import KnowledgeGraph  # noqa: E402
from kg.suggest import find_candidates, suggest_prerequisites  # noqa: E402


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt, system=""):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def real_onto():
    onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml",
                                   ROOT / "data" / "ontology_generated.yaml")
    return onto


def test_find_candidates_are_roots_sorted_by_unlocks():
    onto = real_onto()
    cands = find_candidates(onto, limit=12)
    dg = prerequisite_graph(KnowledgeGraph.from_ontology(onto))
    for c in cands:
        assert not list(dg.predecessors(c["id"])), f"{c['id']} 不该有前置却在候选里"
    counts = [c["unlock_count"] for c in cands]
    assert counts == sorted(counts, reverse=True), "候选应按解锁数降序"


def test_suggest_prerequisites_fake():
    onto = real_onto()
    good = json.dumps({"edges": [
        {"source": "linear_algebra", "target": "attention",
         "confidence": 0.7, "reason": "注意力中的加权求和是线性代数运算"},
    ]}, ensure_ascii=False)
    fake = FakeLLM([good])
    payload, stats = suggest_prerequisites(onto, fake, limit=8)
    assert stats["suggested"] == 1 and stats["candidates"] > 0
    edge = payload["relations"][0]
    assert edge["type"] == "prerequisite_of", "建议边应默认补上关系类型"
    assert "LLM 建议" in edge["note"] and payload["nodes"] == []
    # 提示词应注入候选说明与全节点清单（消歧靠它）
    assert "候选节点" in fake.prompts[0] and "线性代数" in fake.prompts[0]


def test_suggest_empty_edges_is_normal():
    onto = real_onto()
    payload, stats = suggest_prerequisites(onto, FakeLLM([json.dumps({"edges": []})]))
    assert payload["relations"] == [] and stats["suggested"] == 0


def test_suggest_garbage_raises_extract_error():
    onto = real_onto()
    try:
        suggest_prerequisites(onto, FakeLLM(["完全不是 JSON"]))
        raise AssertionError("垃圾输出应抛 ExtractError，由 CLI 统一捕获")
    except ExtractError:
        pass


def test_suggested_edges_survive_ingest_guards():
    """建议边交给 merge_payload 时，会成环的应被拒——LLM 建议不能绕过防线。"""
    onto = real_onto()
    # linear_algebra 已是 attention 的祖先的祖先：直接造一条反向边应被拒
    payload = {"nodes": [], "relations": [
        {"source": "attention", "target": "linear_algebra",
         "type": "prerequisite_of", "confidence": 0.9},
    ]}
    new_onto, rep = merge_payload(onto, payload)
    assert len(rep.rejected) == 1 and "环路" in rep.rejected[0]["reason"]
    assert KnowledgeGraph.from_ontology(new_onto).check_dag() == []


def main() -> int:
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
