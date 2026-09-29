"""LLM 抽取管线测试。直接运行： python tests/test_extract.py（无需 pytest、不打真实 API）。

LLM 用 FakeLLM 注入 —— 管线只依赖 complete(prompt, system) 一个方法，
真实 HTTP 客户端（LLMClient）的构造逻辑单独测，不发网络请求。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg import Node, Ontology, load_ontology_merged, merge_payload  # noqa: E402
from kg.extract import (  # noqa: E402
    SYSTEM_PROMPT,
    _extract_one,
    build_prompt,
    chunk_text,
    existing_node_lines,
    extract_text,
    parse_llm_json,
)
from kg.llm import LLMClient, LLMConfig, LLMConfigError, chat_url, load_config  # noqa: E402

ENV_KEYS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_EMBEDDING_MODEL")


class FakeLLM:
    """按序吐出预置响应，并记录收到的提示词。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        assert system == SYSTEM_PROMPT
        if not self.responses:
            raise AssertionError("FakeLLM 响应用完了，但管线还在调用")
        return self.responses.pop(0)


def _cfg(**kw) -> LLMConfig:
    base = dict(base_url="http://localhost:11434/v1", api_key="", model="fake-model",
                chunk_chars=4000, max_retries=0)
    base.update(kw)
    return LLMConfig(**base)


# ---------------- LLM 客户端与配置 ----------------


def test_chat_url_joining():
    assert chat_url("https://api.x.com/v1") == "https://api.x.com/v1/chat/completions"
    assert chat_url("https://api.x.com") == "https://api.x.com/chat/completions"
    assert chat_url("https://api.x.com/v1/") == "https://api.x.com/v1/chat/completions"
    full = "https://api.x.com/v1/chat/completions"
    assert chat_url(full) == full
    try:
        chat_url("")
        raise AssertionError("空 base_url 应报错")
    except LLMConfigError:
        pass


def test_config_file_then_cli_override(tmp=None):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "c.yaml"
        f.write_text(yaml.safe_dump({
            "base_url": "https://file.example/v1", "api_key": "filekey",
            "model": "file-model", "chunk_chars": 500,
        }), encoding="utf-8")
        cfg = load_config(f)
        assert cfg.model == "file-model" and cfg.api_key == "filekey"
        assert cfg.chunk_chars == 500
        cfg2 = load_config(f, model="cli-model", api_key="clikey")
        assert cfg2.model == "cli-model" and cfg2.api_key == "clikey"
        assert cfg2.base_url == "https://file.example/v1", "CLI 没给的参数不应顶掉文件值"


def test_config_env_and_dollar_interp(tmp=None):
    saved = {k: os.environ.pop(k, None) for k in ENV_KEYS}
    os.environ["MY_SECRET_KEY"] = "envkey123"
    os.environ["LLM_MODEL"] = "env-model"
    os.environ["LLM_EMBEDDING_MODEL"] = "emb-env"
    os.environ["LLM_EMBEDDING_BASE_URL"] = "https://emb.example/v1"
    try:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "c.yaml"
            f.write_text(yaml.safe_dump({
                "base_url": "https://file.example/v1",
                "api_key": "${MY_SECRET_KEY}",
            }), encoding="utf-8")
            cfg = load_config(f)  # 文件没写 model/embedding_model -> 落到环境变量
            assert cfg.model == "env-model"
            assert cfg.api_key == "envkey123", "${VAR} 应被环境变量替换"
            assert cfg.embedding_model == "emb-env", "嵌入模型也应支持环境变量"
            assert cfg.embedding_base_url == "https://emb.example/v1", "嵌入地址也支持环境变量"
            assert cfg.embedding_base_url != cfg.base_url
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        os.environ.pop("MY_SECRET_KEY", None)


def test_config_missing_tells_user_how_to_fix():
    saved = {k: os.environ.pop(k, None) for k in ENV_KEYS}
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                load_config(Path(td) / "nope.yaml")  # 文件、环境变量都没有
                raise AssertionError("缺配置应报 LLMConfigError")
            except LLMConfigError as exc:
                msg = str(exc)
                assert "llm_config.example.yaml" in msg and "--api-key" in msg
                assert "LLM_API_KEY" in msg
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_local_service_allows_empty_api_key():
    LLMConfig(base_url="http://localhost:11434/v1", api_key="", model="llama3").validate()


# ---------------- JSON 容错解析 ----------------


def test_parse_llm_json_variants():
    assert parse_llm_json('{"a": 1}') == {"a": 1}
    assert parse_llm_json('```json\n{"nodes": []}\n```') == {"nodes": []}
    assert parse_llm_json('好的，结果如下：{"nodes": [{"id": "x"}]} 希望有帮助') == {"nodes": [{"id": "x"}]}
    nested = '{"a": {"b": [1, 2]}, "s": "字符串里有 } 和 { 不该干扰"}'
    assert parse_llm_json(f"说明文字 {nested} 结尾") == json.loads(nested)
    try:
        parse_llm_json("完全没有 JSON")
        raise AssertionError("应抛 ExtractError")
    except Exception as exc:
        assert type(exc).__name__ == "ExtractError"


# ---------------- 分片与提示词 ----------------


def test_chunk_text_paragraph_aware():
    assert len(chunk_text("a\n\nb\n\nc", 4000)) == 1
    # chunk_chars 最小钳制为 200（防配置手滑打成海量请求），用 >=200 的分片测真实逻辑
    long_para = "字" * 450
    chunks = chunk_text(long_para, 200)
    assert all(len(c) <= 200 for c in chunks) and "".join(chunks) == long_para
    p1, p2 = "甲" * 150, "乙" * 150
    one = chunk_text(p1 + "\n\n" + p2, 400)          # 两段能拼进一片
    assert len(one) == 1 and one[0].startswith("甲") and one[0].endswith("乙")
    two = chunk_text(p1 + "\n\n" + p2, 200)          # 拼不下 -> 段落边界分两片
    assert len(two) == 2 and two[0] == p1 and two[1] == p2


def test_prompt_carries_existing_nodes_and_chunk():
    onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml")
    prompt = build_prompt("这是关于 MoE 的笔记", existing_node_lines(onto))
    assert "transformer | Transformer" in prompt
    assert "prerequisite_of" in prompt and "confusable_with" in prompt
    assert "这是关于 MoE 的笔记" in prompt
    assert prompt.count("- ") >= 80, "全部已有节点都应注入"


# ---------------- 抽取主流程（FakeLLM） ----------------

GOOD1 = json.dumps({
    "nodes": [{"id": "vit", "name": "Vision Transformer", "type": "Model",
               "domain": "多模态", "difficulty": 4, "aliases": ["ViT"]}],
    "relations": [{"source": "transformer", "target": "vit",
                   "type": "prerequisite_of", "confidence": 0.9}],
}, ensure_ascii=False)
GOOD2 = json.dumps({
    "nodes": [{"id": "vit", "name": "Vision Transformer", "aliases": ["视觉Transformer"]},
              {"id": "clip", "name": "CLIP", "type": "Model", "difficulty": 4}],
    "relations": [{"source": "transformer", "target": "vit",
                   "type": "prerequisite_of", "confidence": 0.7},
                  {"source": "clip", "target": "vit", "type": "related_to", "confidence": 0.8}],
}, ensure_ascii=False)
GARBAGE = "我觉得这段笔记讲得不错，不过没有可以抽取的三元组。"


def test_extract_merges_chunks_and_dedupes():
    fake = FakeLLM([GOOD1, GOOD2])
    onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml")
    text = ("甲" * 150) + " ViT\n\n" + ("乙" * 150) + " CLIP"
    payload, stats = extract_text(text, fake, _cfg(chunk_chars=200), onto)
    assert stats["chunks"] == 2 and stats["ok_chunks"] == 2
    assert {n["id"] for n in payload["nodes"]} == {"vit", "clip"}
    vit = next(n for n in payload["nodes"] if n["id"] == "vit")
    assert set(vit["aliases"]) == {"ViT", "视觉Transformer"}, "跨片段别名应并集"
    # 同 key 关系保最高置信度：0.9 保留，0.7 丢弃
    assert len(payload["relations"]) == 2
    kept = next(r for r in payload["relations"]
                if (r["source"], r["type"]) == ("transformer", "prerequisite_of"))
    assert kept["confidence"] == 0.9

    # 再过一遍真正的入库合并：transformer 已存在 -> 报告应为纯新增
    _new_onto, rep = merge_payload(onto, payload)
    assert rep.added_nodes == ["vit", "clip"]
    assert rep.rejected == []
    assert len(rep.added_relations) == 2


def test_extracted_sources_keep_only_verified_quotes():
    onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml")
    response = json.dumps({
        "nodes": [{"id": "new_attention", "name": "新注意力", "type": "Concept",
                   "evidence_quote": "注意力会聚合上下文"}],
        "relations": [{"source": "attention", "target": "new_attention",
                       "type": "related_to", "evidence_quote": "原文里没有的句子"}],
    }, ensure_ascii=False)
    payload, _ = extract_text("注意力会聚合上下文。", FakeLLM([response]),
                              _cfg(), onto, document="private/notes.md")
    node_ref = payload["nodes"][0]["evidence_refs"][0]
    rel_ref = payload["relations"][0]["evidence_refs"][0]
    assert node_ref["document"] == "notes.md" and node_ref["chunk_id"] == "chunk-0000"
    assert node_ref["quote_verified"] and node_ref["quote"] == "注意力会聚合上下文"
    assert not rel_ref["quote_verified"] and rel_ref["quote"] == "", \
        "模型编造的引文不可当作原文引用"


def test_extract_two_chunks_when_small_limit():
    fake = FakeLLM([GOOD1, GOOD2])
    onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml")
    _payload, stats = extract_text("甲" * 150 + "\n\n" + "乙" * 150,
                                   fake, _cfg(chunk_chars=200), onto)
    assert stats["chunks"] == 2
    assert fake.prompts[0] != fake.prompts[1], "两个分片应各自发一次请求"


def test_extract_retry_on_garbage_then_success():
    fake = FakeLLM([GARBAGE, GOOD1])
    onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml")
    payload, stats = extract_text("一篇笔记", fake, _cfg(max_retries=1), onto)
    assert stats["ok_chunks"] == 1 and stats["nodes"] == 1
    assert "无法解析" in fake.prompts[1], "重试时应附加纠错指令"


def test_extract_failed_chunk_does_not_kill_batch():
    fake = FakeLLM([GARBAGE, GOOD2])
    onto, _ = load_ontology_merged(ROOT / "data" / "seed_ontology.yaml")
    payload, stats = extract_text("甲" * 150 + "\n\n" + "乙" * 150,
                                  fake, _cfg(chunk_chars=200, max_retries=0), onto)
    assert len(stats["failed_chunks"]) == 1
    assert stats["ok_chunks"] == 1 and {n["id"] for n in payload["nodes"]} == {"vit", "clip"}


def test_extract_one_raises_without_retry_budget():
    fake = FakeLLM([GARBAGE])
    try:
        _extract_one(fake, "prompt", max_retries=0)
        raise AssertionError("无重试额度且解析失败应抛错")
    except Exception as exc:
        assert type(exc).__name__ == "ExtractError"


def test_ontology_untouched_by_helper(tmp=None):
    """existing_node_lines 是纯函数，不应修改本体。"""
    onto = Ontology(nodes=[Node(id="a", name="甲", type="Concept")], relations=[])
    _lines = existing_node_lines(onto)
    assert len(onto.nodes) == 1


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
