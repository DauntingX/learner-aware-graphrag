"""MCP Server 测试。直接运行： python tests/test_mcp.py（无需 pytest）。

两部分：
  1. 工具逻辑直测（update_mastery 的掌握度数学用临时文件，不动真实画像）
  2. stdio 协议回环：真实拉起 mcp_server.py 子进程，走完整 MCP 握手与工具调用
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mcp_server  # noqa: E402

EXPECTED_TOOLS = {
    "get_learner_profile",
    "check_readiness",
    "get_learning_path",
    "recommend_next",
    "query_node",
    "list_domains",
    "update_mastery",
    "search_knowledge",
}


# ---------------- 1. 工具逻辑直测 ----------------


def test_update_mastery_math(tmp_learner):
    mcp_server.LEARNER_PATH = tmp_learner
    r1 = mcp_server.update_mastery("transformer", "quiz", "答对一道注意力题", 0.6)
    assert abs(r1["new_mastery"] - 0.6) < 1e-3, r1
    r2 = mcp_server.update_mastery("transformer", "project", "实现了一个 mini Transformer", 0.6)
    # 1 - (1-0.6)*(1-0.6) = 0.84，今日记录不衰减
    assert abs(r2["new_mastery"] - 0.84) < 1e-3, r2
    assert r2["evidence_count"] == 2


def test_missing_private_profile_starts_empty(tmp_learner):
    mcp_server.LEARNER_PATH = tmp_learner
    assert not tmp_learner.exists()
    profile = mcp_server.get_learner_profile()
    assert profile["user_id"] == "me"
    result = mcp_server.update_mastery("transformer", "quiz", "首次答题", 0.6)
    assert result["evidence_count"] == 1
    saved = json.loads(tmp_learner.read_text(encoding="utf-8"))
    assert saved["user_id"] == "me"
    assert set(saved["states"]) == {"transformer"}


def test_update_mastery_synthesizes_history_evidence(tmp_learner):
    # 历史状态只有总分没有证据明细：回写时应折算成等效证据，不能丢掉旧掌握度
    previous = (date.today() - timedelta(days=30)).isoformat()
    tmp_learner.write_text(json.dumps({
        "user_id": "me",
        "states": {"attention": {"mastery": 0.8, "confidence": 0.7,
                                 "last_touched": previous, "source": "self_report"}},
    }, ensure_ascii=False), encoding="utf-8")
    mcp_server.LEARNER_PATH = tmp_learner
    r = mcp_server.update_mastery("attention", "conversation", "用户自己讲清了 QKV", 0.4)
    assert 0.4 < r["new_mastery"] < 0.88, r  # 历史证据先衰减，新证据为今天
    assert r["evidence_count"] == 2
    saved = json.loads(tmp_learner.read_text(encoding="utf-8"))
    assert saved["states"]["attention"]["evidence"][0]["ts"] == previous


def test_update_mastery_rejects_bad_input(tmp_learner):
    mcp_server.LEARNER_PATH = tmp_learner
    assert "error" in mcp_server.update_mastery("不存在的概念xyz")
    assert "error" in mcp_server.update_mastery("transformer", evidence_type="魔法证据")


def test_read_tools_smoke(tmp_learner):
    mcp_server.LEARNER_PATH = tmp_learner
    prof = mcp_server.get_learner_profile()
    assert prof["total_concepts"] > 0
    ready = mcp_server.check_readiness("graph_rag")
    assert ready["verdict"] in {"ready", "needs_review", "not_ready"}
    assert ready["prerequisites_total"] > 0
    path = mcp_server.get_learning_path("graph_rag")
    assert path["total"] > 0 and path["steps"][0]["order"] == 0
    node = mcp_server.query_node("Transformer")
    assert node["name"] == "Transformer"
    doms = mcp_server.list_domains()
    assert len(doms["domains"]) > 0


# ---------------- 2. stdio 协议回环 ----------------


def test_mcp_stdio_roundtrip():
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_server.py")],
            cwd=str(ROOT),
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert EXPECTED_TOOLS <= names, f"缺工具: {EXPECTED_TOOLS - names}"

                async def call(name, args):
                    res = await session.call_tool(name, args)
                    assert not res.is_error, res.content
                    return json.loads(res.content[0].text)

                prof = await call("get_learner_profile", {})
                assert prof["total_concepts"] > 0

                ready = await call("check_readiness", {"target": "graph_rag"})
                assert ready["target"] == "graph_rag"
                assert ready["verdict"] in {"ready", "needs_review", "not_ready"}

                path = await call("get_learning_path", {"target": "graph_rag", "include_known": False})
                assert path["total"] > 0

                rec = await call("recommend_next", {"limit": 3})
                assert len(rec["recommendations"]) <= 3

                node = await call("query_node", {"keyword": "Transformer"})
                assert node["name"] == "Transformer"

                rag = await call("search_knowledge", {"query": "KV 缓存怎么优化"})
                assert rag["mode"] == "local" and rag["context"]
                assert any(n["role"] == "prereq" for n in rag["context"]), \
                    "GraphRAG 检索应扩展出前置知识"

                global_rag = await call("search_knowledge", {
                    "query": "评估方法领域整体有哪些知识", "mode": "global"})
                assert global_rag["mode"] == "global"
                assert global_rag["context"] and "社区 #" in global_rag["context_text"]

    anyio.run(run)


def test_update_mastery_revocation(tmp_learner):
    """graphiti 式证据可撤销：撤销最近一条证据后重算，轨迹保留但不生效。"""
    from kg.schema import Evidence, compute_mastery

    # schema 层：revoked 证据不参与计算（不给 last_touched，避免时间衰减干扰断言）
    evs = [Evidence(type="quiz", desc="a", weight=0.6),
           Evidence(type="quiz", desc="b", weight=0.6, revoked=True)]
    assert abs(compute_mastery(evs) - 0.6) < 1e-3

    mcp_server.LEARNER_PATH = tmp_learner
    mcp_server.update_mastery("transformer", "quiz", "第一次作答", 0.6)
    r2 = mcp_server.update_mastery("transformer", "project", "第二次作答", 0.6)
    assert abs(r2["new_mastery"] - 0.84) < 1e-3
    r3 = mcp_server.update_mastery("transformer", revoke_last=True)
    assert abs(r3["new_mastery"] - 0.6) < 1e-3, "撤销第二次后应回到第一次的水平"
    assert "第二次" in r3["revoked"]
    r4 = mcp_server.update_mastery("transformer", revoke_last=True)
    assert abs(r4["new_mastery"]) < 1e-3, "全部撤销后掌握度归零"
    assert "error" in mcp_server.update_mastery("transformer", revoke_last=True), \
        "没有可撤销证据时应报错而不是静默"


def main() -> int:
    import inspect
    import traceback

    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        needs_tmp = "tmp_learner" in inspect.signature(fn).parameters
        try:
            if needs_tmp:
                # 用临时画像文件跑，不动真实 data/learner_state.json
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td) / "learner_state.json"
                    old = mcp_server.LEARNER_PATH
                    mcp_server.LEARNER_PATH = tmp
                    try:
                        fn(tmp)
                    finally:
                        mcp_server.LEARNER_PATH = old
            else:
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
