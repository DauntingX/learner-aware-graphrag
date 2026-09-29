"""笔记解析与抽取测试。直接运行： python tests/test_notes.py（无需 pytest、不打真实 API）。

覆盖：md/docx 结构解析（标题栈/面包屑/代码围栏）、小节分片（面包屑前缀/超长细切）、
FakeLLM 端到端笔记入库、GraphRAG 式 gleaning 补抽。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg import load_ontology_merged, merge_payload  # noqa: E402
from kg.extract import (  # noqa: E402
    _extract_one,
    extract_note,
    sections_to_chunks,
)
from kg.llm import LLMConfig  # noqa: E402
from kg.notes import read_note, parse_markdown, parse_docx  # noqa: E402


class FakeLLM:
    """按序吐出预置响应，并记录收到的提示词。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeLLM 响应用完了，但管线还在调用")
        return self.responses.pop(0)


def _cfg(**kw) -> LLMConfig:
    base = dict(base_url="http://localhost:11434/v1", api_key="", model="fake-model",
                chunk_chars=4000, max_retries=0)
    base.update(kw)
    return LLMConfig(**base)


MD_SAMPLE = """开头没有标题的导语。

# Agent 笔记

## 记忆系统

短期记忆是上下文窗口。

### 长期记忆

长期记忆用向量库。

```python
# 这里的 # 不是标题
def f(): pass
```

## 工具调用

function calling 输出结构化参数。
"""


def test_parse_markdown_headings_and_breadcrumbs():
    secs = parse_markdown(MD_SAMPLE)
    titles = [(s.title, s.breadcrumb, s.level) for s in secs]
    assert ("正文", "正文", 0) in titles, "标题前的导语应单独成节"
    assert ("记忆系统", "Agent 笔记 > 记忆系统", 2) in titles
    assert ("长期记忆", "Agent 笔记 > 记忆系统 > 长期记忆", 3) in titles
    assert ("工具调用", "Agent 笔记 > 工具调用", 2) in titles
    # 代码块里的 # 不应被当成标题
    assert not any("这里的" in s.title for s in secs)
    memory = next(s for s in secs if s.title == "长期记忆")
    assert "# 这里的 # 不是标题" in memory.text, "围栏内容应完整保留在正文里"


def test_parse_markdown_flat_and_no_heading():
    assert len(parse_markdown("没有标题只有正文。\n\n第二段。")) == 1
    flat = parse_markdown("# A\n\ntext a\n\n# B\n\ntext b")
    assert [s.title for s in flat] == ["A", "B"], "同级标题应并列，不互相嵌套"


def test_parse_docx(tmp=None):
    import docx

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.docx"
        doc = docx.Document()
        doc.add_heading("学习笔记", 1)
        doc.add_paragraph("RAG 是检索增强生成。")
        doc.add_heading("检索部分", 2)
        doc.add_paragraph("向量检索用余弦相似度。")
        doc.save(str(p))
        secs = parse_docx(p)
        assert [(s.title, s.breadcrumb) for s in secs] == [
            ("学习笔记", "学习笔记"),
            ("检索部分", "学习笔记 > 检索部分"),
        ]
        assert "余弦相似度" in secs[1].text
        # read_note 按扩展名分发
        assert len(read_note(p)) == 2


def test_sections_to_chunks_breadcrumb_and_oversize():
    secs = parse_markdown(MD_SAMPLE)
    chunks = sections_to_chunks(secs, chunk_chars=4000)
    assert all(c.startswith("【笔记位置：") for c in chunks), "每片应带面包屑前缀"
    assert any("长期记忆" in c.split("\n")[0] for c in chunks)
    # 超长小节细切后每片仍带面包屑
    big = parse_markdown("# 大节\n\n" + ("很长的段落。 " * 300))
    sub = sections_to_chunks(big, chunk_chars=300)
    assert len(sub) > 1 and all(c.startswith("【笔记位置：大节】") for c in sub)


def test_extract_note_end_to_end_fake(tmp=None):
    GOOD = json.dumps({
        "nodes": [{"id": "agent_memory", "name": "记忆系统", "type": "Technique",
                   "domain": "智能体Agent", "difficulty": 3}],
        "relations": [{"source": "agent_memory", "target": "rag",
                       "type": "prerequisite_of", "confidence": 0.8}],
    }, ensure_ascii=False)
    with tempfile.TemporaryDirectory() as td:
        note = Path(td) / "n.md"
        note.write_text(MD_SAMPLE, encoding="utf-8")
        fake = FakeLLM([GOOD, GOOD, GOOD, GOOD, GOOD])  # 导语 + 3 个有正文的小节（gleaning 关）
        onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml",
                                       ROOT / "data" / "ontology_generated.yaml")
        payload, stats = extract_note(str(note), fake, _cfg(), onto)
        assert stats["ok_chunks"] == stats["chunks"] and stats["nodes"] == 1
        assert any("笔记位置" in p and "长期记忆" in p for p in fake.prompts), \
            "提示词里应注入面包屑"
        _new_onto, rep = merge_payload(onto, payload)
        assert rep.added_nodes == ["agent_memory"] and rep.rejected == []


def test_gleaning_merges_supplement():
    FIRST = json.dumps({"nodes": [{"id": "a", "name": "概念A", "type": "Concept"}],
                        "relations": []}, ensure_ascii=False)
    EXTRA = json.dumps({"nodes": [{"id": "b", "name": "概念B", "type": "Concept"}],
                        "relations": [{"source": "b", "target": "a",
                                       "type": "related_to", "confidence": 0.7}]},
                       ensure_ascii=False)
    fake = FakeLLM([FIRST, EXTRA, EXTRA])
    payload = _extract_one(fake, "首抽提示词", max_retries=0, gleans=2)
    ids = {n["id"] for n in payload["nodes"]}
    assert ids == {"a", "b"}, "补抽结果应并入首抽"
    assert len(payload["relations"]) == 1
    assert "遗漏" in fake.prompts[1], "第二轮应是 gleaning 追问"
    # gleans=0 时不追问
    fake2 = FakeLLM([FIRST])
    p2 = _extract_one(fake2, "提示词", max_retries=0, gleans=0)
    assert {n["id"] for n in p2["nodes"]} == {"a"} and len(fake2.prompts) == 1


def test_gleaning_failure_does_not_hurt_first_round():
    FIRST = json.dumps({"nodes": [{"id": "a", "name": "概念A", "type": "Concept"}],
                        "relations": []}, ensure_ascii=False)
    fake = FakeLLM([FIRST, "补抽输出完全不是 JSON"])
    payload = _extract_one(fake, "提示词", max_retries=0, gleans=1)
    assert {n["id"] for n in payload["nodes"]} == {"a"}, "补抽失败只放弃补充，不损首轮"


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
