# LearnerGraph：学习者感知的 AI 知识图谱

把 AI 知识点、前置依赖和学习状态放在同一张图中，为检索、学习路径规划和 Agent 工具调用提供依据。项目可离线运行；LLM 抽取、语义嵌入与生成式回答按需接入。

**从这里开始：** [2 分钟运行](#2-分钟运行) · [公开演示与隐私](#公开演示与隐私) · [检索评测](#检索评测) · [设计与边界](#设计与边界)

> 仓库中的学习画像是**完全合成的演示数据**。真实画像、笔记和由它们生成的索引不会进入公开演示。

## 项目能做什么

| 能力 | 当前实现 |
|---|---|
| 知识结构 | 人工维护的种子本体：88 个节点、117 条关系；前置关系形成有向无环图 |
| 笔记入库 | Markdown / DOCX 按标题小节切分，LLM 抽取候选节点与关系，经消歧、模式校验、去重和成环检查后进入独立补丁层 |
| 学习者模型 | 掌握度、证据与置信度分离；学习路径按前置顺序排，并标出已掌握、学习中和缺口 |
| 检索 | 本地查询从向量候选出发扩展图邻居；全局查询按 Louvain 社区摘要检索；上下文可带掌握度与来源信息 |
| Agent 接口 | MCP 服务提供画像、就绪检查、路径、推荐、图查询、领域概览、掌握证据回写和图检索 |
| 可视化 | D3 力导向图、筛选、搜索、节点详情和学习路径播放；可构建纯静态公开页面 |

三层数据关系：

```text
笔记 / 种子本体 ──> 知识图谱（知识点、关系、前置链）
                           │
合成或本地画像 ────────────┤──> 个性化检索 / 学习路径 / MCP 工具
                           └──> 浏览器可视化
```

## 2 分钟运行

要求 Python 3.11+。以下命令在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python build_graph.py
python serve.py
```

打开 `http://127.0.0.1:8000`。全新克隆没有私人 `data/learner_state.json` 时，构建会自动使用 `data/learner_state.example.json`；如果本机存在私人文件，则优先使用它。普通构建会使用本机机器补丁层，并把网页数据写入已忽略的 `web/graph_data.json` 和 `web/standalone.html`。

几个不需要 API 密钥的入口：

```bash
python build_graph.py --path graph_rag --profile
python ingest.py data/examples/triples_demo.json --dry-run
python graph_rag.py "GraphRAG 是什么"
python graph_rag.py "评估方法领域都有什么" --global
python graph_rag.py --compare
```

示例三元组位于 `data/examples/triples_demo.json`。去掉 `--dry-run` 才会写入 `data/ontology_generated.yaml`；该补丁层属于本地数据，不提交到公开仓库。

## 公开演示与隐私

```bash
python build_graph.py --public-demo
python -m http.server 8001 --directory dist/public-demo
```

打开 `http://127.0.0.1:8001`。`--public-demo` **强制只读种子本体和合成画像**，将网页所需的白名单资源输出到 `dist/public-demo/`。即使本机有私人学习画像和由笔记生成的图谱补丁，公开构建也不会读取它们；演示页会明确标示合成数据。

`.github/workflows/pages.yml` 提供手动触发的 GitHub Pages 发布流程；仓库创建后需在 Pages 设置中选择 GitHub Actions 作为发布源，再手动运行该工作流。CI 也会构建并核对公开产物的文件清单和合成数据标记。Pages 发布流程不会上传整个工作目录。

以下内容由 `.gitignore` 排除：本地密钥配置、真实画像及备份、笔记与抽取收件箱、机器补丁层、SQLite、向量和社区索引、网页构建物。公开前仍应核对待提交文件，特别是截图、额外语料和手工新增的文件。

## 检索评测

`python graph_rag.py --compare` 使用仓库中的自建查询集，对比朴素向量召回与图扩展检索，输出目标命中、前置链覆盖、上下文规模等指标。默认使用确定性的本地字符 n-gram 哈希嵌入，便于离线复现；配置语义嵌入后需重建索引并单独记录结果。

当前查询集规模较小。默认对比让两种检索各使用最多 6 个节点，并同时输出实际上下文规模；用 `--max-context` 改动图检索预算时，应按相同节点或 token 预算解读结果。这个评测用于发现前置知识遗漏和回归，不代表公开基准上的通用 RAG 排名。

## 设计与边界

- **种子只读，候选需审核。** 手写本体保留注释与稳定 ID；机器抽取写入独立补丁层。LLM 的输出先过确定性校验，前置边成环时逐条拒绝。
- **查询意图优先。** 本地检索先按查询相似度保留核心种子，再用前置关系和掌握度填充教学上下文。掌握度不会把用户明确询问的节点挤出种子。
- **可追溯的候选来源。** 抽取与入库可记录文档、分片和标题位置；只有与原文精确匹配的短引文才作为可验证引文保留。这是分片级来源追踪，尚非完整的文档引用系统。
- **可撤销的学习证据。** 证据记录可以标记失效并保留轨迹；读取时按当前时间计算掌握度，避免把旧的存储分数当成永不变化的结论。
- **全局检索边界清楚。** 当前是单层 Louvain 社区和抽取式摘要，不生成 Microsoft GraphRAG 式的层级社区报告。
- **小规模、低门槛。** 图使用 NetworkX 与 SQLite；向量索引存 JSON。适合个人知识库和可复现实验，尚未按大规模生产负载优化。

实现参考与差异：[Microsoft GraphRAG](https://microsoft.github.io/graphrag/) 的 local/global 检索、[LightRAG](https://github.com/HKUDS/LightRAG) 的轻量图检索、[Graphiti](https://github.com/getzep/graphiti) 的时序证据思想，以及 [MCP 官方文档](https://modelcontextprotocol.io/)。本项目重点是教学前置链与学习者状态如何影响 Agent 可用的上下文。

## 接入 LLM 与 MCP

不配置 LLM 时，建图、路径、MCP 基础工具和离线检索都可用。抽取笔记、语义嵌入和生成式回答需要配置兼容接口：

```bash
# 复制模板并在本地填写；llm_config.yaml 已被忽略
cp llm_config.example.yaml llm_config.yaml
python extract.py data/corpus/demo_notes.md --dry-run
python graph_rag.py "RLHF 怎么训练" --answer
```

Windows PowerShell 可用 `Copy-Item llm_config.example.yaml llm_config.yaml`。密钥也可放在环境变量中；不要写入示例文件、截图或提交内容。

MCP 客户端可把 `mcp_server.py` 作为 stdio 服务启动。例如把下方路径替换为本机仓库的绝对路径：

```json
{
  "mcpServers": {
    "learner-graph": {
      "command": "python",
      "args": ["/absolute/path/to/tupu/mcp_server.py"]
    }
  }
}
```

真实证据回写会修改本地画像。公开演示仅展示合成画像，不提供在线写入接口。

## 开发验证

CI 在 Python 3.11 与 3.13 上运行各模块测试，并构建公开演示。也可逐个运行：

```bash
python tests/test_ingest.py
python tests/test_extract.py
python tests/test_notes.py
python tests/test_rag.py
python tests/test_suggest.py
python tests/test_learner.py
python tests/test_mcp.py
```

主要代码位于 `src/kg/`：`schema.py` 管数据约束，`ingest.py` 管合并与校验，`learner.py` 管画像，`paths.py` 管路径，`rag.py` 管检索，`extract.py` / `notes.py` 管笔记抽取。`web/` 是无后端的演示界面。
