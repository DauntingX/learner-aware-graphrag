#!/usr/bin/env python
"""一键构建知识图谱。

流程：
    seed_ontology.yaml  ->  NetworkX 图  ->  自检  ->  SQLite  ->  前端 JSON

用法：
    python build_graph.py                # 完整构建 + 导出
    python build_graph.py --path rag     # 额外打印一条到 target 的学习路径
    python build_graph.py --profile      # 额外打印知识画像
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kg import KnowledgeGraph, LearnerModel, load_ontology_merged, learning_path, recommend_next  # noqa: E402

DATA = ROOT / "data"
WEB = ROOT / "web"
PUBLIC_DEMO = ROOT / "dist" / "public-demo"
PUBLIC_ASSETS = (
    "index.html", "styles.css", "app.js", "vendor/d3.min.js", "vendor/LICENSE-d3.txt",
)


def banner(title: str) -> None:
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


def bundle_standalone(web_dir: Path = WEB) -> Path | None:
    """把 HTML / CSS / D3 / 应用脚本 / 图数据打成一个自包含文件。

    用途：双击就能看，不需要起服务器（浏览器不允许 file:// 页面 fetch 本地 JSON）。
    演示、发给别人看、写进简历附件都用它。
    """
    idx = web_dir / "index.html"
    if not idx.exists():
        return None

    html = idx.read_text(encoding="utf-8")
    css = (web_dir / "styles.css").read_text(encoding="utf-8")
    d3js = (web_dir / "vendor" / "d3.min.js").read_text(encoding="utf-8")
    d3_license = (web_dir / "vendor" / "LICENSE-d3.txt").read_text(encoding="utf-8")
    app = (web_dir / "app.js").read_text(encoding="utf-8")
    data = (web_dir / "graph_data.json").read_text(encoding="utf-8")

    # 内联脚本里若出现 </script> 会提前闭合标签，必须转义
    d3js = d3js.replace("</script>", "<\\/script>")
    app = app.replace("</script>", "<\\/script>")
    data = data.replace("<", "\\u003c")

    html = html.replace(
        '<link rel="stylesheet" href="styles.css">', f"<style>\n{css}\n</style>"
    )
    html = html.replace(
        '<script src="vendor/d3.min.js"></script>',
        f"<script>\n/*\n{d3_license}\n*/\n{d3js}\n</script>",
    )
    html = html.replace('<script src="app.js"></script>', f"<script>\n{app}\n</script>")
    html = html.replace(
        '<script id="kg-data" type="application/json"></script>',
        f'<script id="kg-data" type="application/json">{data}</script>',
    )

    out = web_dir / "standalone.html"
    out.write_text(html, encoding="utf-8")
    return out


def prepare_public_demo() -> Path:
    """只复制公开资源，拒绝把其它本地文件混入 Pages 产物。"""
    allowed = set(PUBLIC_ASSETS) | {"graph_data.json", "standalone.html"}
    if PUBLIC_DEMO.exists():
        unexpected = [
            p.relative_to(PUBLIC_DEMO).as_posix()
            for p in PUBLIC_DEMO.rglob("*")
            if (p.is_file() or p.is_symlink())
            and (p.is_symlink() or p.relative_to(PUBLIC_DEMO).as_posix() not in allowed)
        ]
        if unexpected:
            raise RuntimeError(f"公开演示目录含非白名单文件，请人工检查：{unexpected}")
    PUBLIC_DEMO.mkdir(parents=True, exist_ok=True)
    for asset in PUBLIC_ASSETS:
        source = WEB / asset
        target = PUBLIC_DEMO / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return PUBLIC_DEMO


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 AI 知识图谱")
    ap.add_argument("--path", help="计算到某个知识点（id 或名称）的学习路径")
    ap.add_argument("--profile", action="store_true", help="打印知识画像")
    ap.add_argument("--no-web", action="store_true", help="跳过前端 JSON 导出")
    ap.add_argument("--public-demo", action="store_true", help="只用种子本体与合成画像构建 Pages 演示")
    args = ap.parse_args()
    if args.public_demo and args.no_web:
        ap.error("--public-demo 需要导出网页，不能与 --no-web 同用")

    # ---------- 1. 加载本体并构图 ----------
    banner("1 / 5  加载种子本体")
    seed_path = DATA / "seed_ontology.yaml"
    sources = [seed_path] if args.public_demo else [seed_path, DATA / "ontology_generated.yaml"]
    onto, loaded = load_ontology_merged(*sources)
    kg = KnowledgeGraph.from_ontology(onto)
    print(f"  基底：{seed_path.relative_to(ROOT)}")
    if len(loaded) > 1:
        print(f"  已叠加机器补丁层：{loaded[1].relative_to(ROOT)}（P1 入库产物）")
    print(f"  节点 {len(kg.g)} 个，关系 {kg.g.number_of_edges()} 条")

    # ---------- 2. 自检 ----------
    banner("2 / 5  图谱自检")
    stats = kg.stats()
    print("  节点类型：", "  ".join(f"{k}×{v}" for k, v in sorted(stats["node_types"].items())))
    print("  关系类型：")
    for k, v in sorted(stats["relation_types"].items(), key=lambda x: -x[1]):
        print(f"      {k:<18} {v}")
    print(f"  领域分布：{len(stats['domains'])} 个")
    for k, v in sorted(stats["domains"].items(), key=lambda x: -x[1]):
        print(f"      {k:<14} {v}")

    cycles = kg.check_dag()
    if cycles:
        print(f"  [FAIL] prerequisite_of 存在环路 {len(cycles)} 处：{cycles[:3]}")
        return 1
    print("  [OK] prerequisite_of 构成 DAG，可拓扑排序")

    multi = kg.multi_relation_pairs()
    print(f"  [INFO] 多重关系对（同一对节点并存多种关系）{len(multi)} 组：")
    for pair, types in list(multi.items())[:4]:
        print(f"        {pair}   {' + '.join(types)}")
    if len(multi) > 4:
        print(f"        ... 其余 {len(multi) - 4} 组")

    isolated = stats["isolated"]
    print(f"  [{'OK' if not isolated else 'WARN'}] 孤立节点：{len(isolated)} 个")
    if isolated:
        print(f"        {isolated}")

    roots = kg.roots()
    print(f"  [INFO] 学习起点（无前置）{len(roots)} 个：{roots[:6]}{' ...' if len(roots) > 6 else ''}")
    print(f"  [INFO] 阶段性终点（无后继）{len(kg.leaves())} 个")

    # ---------- 3. 持久化 ----------
    banner("3 / 5  持久化到 SQLite")
    db_context = tempfile.TemporaryDirectory() if args.public_demo else nullcontext(DATA)
    with db_context as db_dir:
        db = Path(db_dir) / "kg.sqlite"
        kg.save_sqlite(db)
        reloaded = KnowledgeGraph.from_sqlite(db)
        assert reloaded.g.number_of_nodes() == kg.g.number_of_nodes(), "SQLite 回读节点数不一致"
        assert reloaded.g.number_of_edges() == kg.g.number_of_edges(), "SQLite 回读边数不一致"
        db_label = "临时 SQLite" if args.public_demo else str(db.relative_to(ROOT))
        print(f"  [OK] {db_label} 写入并回读校验通过")

    # ---------- 4. 学习者模型 ----------
    banner("4 / 5  加载学习者状态")
    private_learner_path = DATA / "learner_state.json"
    demo_learner_path = DATA / "learner_state.example.json"
    use_demo_learner = args.public_demo or not private_learner_path.exists()
    learner_path = demo_learner_path if use_demo_learner else private_learner_path
    learner = LearnerModel.load(learner_path)
    covered = len(learner.states)
    avg = (
        sum(learner.mastery_or_zero(cid) for cid in learner.states) / covered if covered else 0.0
    )
    print(f"  [OK] {learner_path.name}  已记录 {covered}/{len(kg.g)} 个知识点，平均掌握度 {avg:.2f}")

    if args.profile:
        print()
        ids = sorted(learner.states.keys())
        prof = learner.profile(ids)
        for band, items in prof["bands"].items():
            if items:
                names = [kg.node(i).name for i in items if kg.node(i)]
                print(f"    {band:<10} {len(items):>3}  {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")

    # ---------- 5. 学习路径演示 ----------
    if args.path:
        banner(f"5 / 5  学习路径：-> {args.path}")
        hits = kg.find(args.path)
        if not hits:
            print(f"  [FAIL] 没找到匹配 '{args.path}' 的知识点")
            return 1
        target = hits[0].id
        result = learning_path(kg, target, learner)
        print(f"  目标：{result['target_name']}   共 {result['total']} 步，其中缺口 {result['gap_count']} 个\n")
        for s in result["steps"]:
            mark = {"gap": "[缺]", "learning": "[学]", "known": "[会]"}[s["status"]]
            print(
                f"   {mark} {s['order'] + 1:>2}. {s['name']:<22} "
                f"掌握度 {s['mastery']:.2f}  难度 {s['difficulty']}  {s['domain']}"
            )

    # ---------- 导出前端数据 ----------
    if not args.no_web:
        banner("导出前端数据")
        output_dir = prepare_public_demo() if args.public_demo else WEB
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = kg.to_payload(learner_getter=learner.get_mastery)
        payload["meta"] = {
            "demo": use_demo_learner,
            "stats": {
                "nodes": stats["nodes"],
                "edges": stats["edges"],
                "domains": stats["domains"],
            },
            "domains": sorted(stats["domains"].keys()),
            "relation_types": sorted(stats["relation_types"].keys()),
        }
        out = output_dir / "graph_data.json"
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"  [OK] {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")

        standalone = bundle_standalone(output_dir)
        if standalone:
            print(
                f"  [OK] {standalone.relative_to(ROOT)}  "
                f"({standalone.stat().st_size / 1024:.0f} KB)  双击即可查看，无需服务器"
            )

        # 顺带导出几条推荐
        rec = recommend_next(kg, learner, limit=5)
        if rec:
            print("\n  现在最该学的 5 个知识点（前置已满足）：")
            for r in rec:
                print(f"      · {r['name']:<18} 难度 {r['difficulty']}  解锁 {r['unlocks']} 个后续")

    if args.public_demo:
        print(f"\n公开演示已生成：{PUBLIC_DEMO.relative_to(ROOT)}（仅含合成数据）\n")
    else:
        print("\n构建完成。启动可视化： python serve.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
