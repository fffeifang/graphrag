# GraphRAG 技术深度分析报告

## 1. 项目概述

GraphRAG 是微软研究院开发的**图增强检索生成（Graph-based RAG）**系统，当前版本 3.0.6。

**核心问题：** 传统 RAG 只能基于向量相似度做局部检索，无法回答需要跨文档综合推理的"全局"查询（如："这些文档的主要主题是什么？"）。GraphRAG 的创新在于：**索引阶段用 LLM 将文本转化为结构化知识图谱（实体 + 关系 + 社区），并预计算多层次摘要，从而支持跨文档全局推理。**

---

## 2. 整体架构

### 2.1 Monorepo 包结构（3.0.0 重大重构）

```
graphrag/packages/
├── graphrag/           ← 主包（CLI + 核心逻辑）
├── graphrag-llm/       ← LLM 抽象层（completion + embedding）
├── graphrag-storage/   ← 存储后端抽象
├── graphrag-vectors/   ← 向量存储抽象
├── graphrag-cache/     ← LLM 响应缓存
├── graphrag-chunking/  ← 文本分块
├── graphrag-input/     ← 文档输入读取
└── graphrag-common/    ← 共享基础设施（Factory 模式）
```

将原本单一包拆分为 8 个独立可发布子包，每个子包可单独安装使用。

### 2.2 主包内部结构

```
packages/graphrag/graphrag/
├── api/                 ← 公开 Python API（build_index / search）
├── callbacks/           ← 管道事件回调（WorkflowCallbacks, QueryCallbacks）
├── cli/                 ← 命令行入口（index, query, prompt_tune, init）
├── config/              ← 全局配置模型（GraphRagConfig）
├── data_model/          ← 核心数据模型（Entity, Relationship, Community 等）
├── graphs/              ← 图算法（Leiden 社区检测、LCC、degree 计算）
├── index/               ← 索引流水线（核心）
│   ├── run/             ← 流水线运行器
│   ├── workflows/       ← 每个处理阶段（一个 workflow function）
│   ├── operations/      ← 底层操作算子
│   └── typing/          ← 类型定义（Pipeline, Workflow, Context 等）
├── prompt_tune/         ← Prompt 自动调优工具
├── prompts/             ← 所有 LLM 提示词模板
├── query/               ← 查询引擎
└── tokenizer/           ← 分词器工具
```

---

## 3. Pipeline/流程设计（Placement 重点）

### 3.1 核心类型体系

```python
# 最基础的类型
WorkflowFunction = Callable[[GraphRagConfig, PipelineRunContext], Awaitable[WorkflowFunctionOutput]]
Workflow = tuple[str, WorkflowFunction]   # (名称, 函数)

# 每次运行的共享上下文
@dataclass
class PipelineRunContext:
    stats: PipelineRunStats
    input_storage: Storage
    output_storage: Storage
    output_table_provider: TableProvider
    previous_table_provider: TableProvider | None  # 增量更新时的前次输出
    cache: Cache
    callbacks: WorkflowCallbacks
    state: PipelineState                           # 运行时任意状态字典
```

### 3.2 PipelineFactory — 注册与创建中枢

```python
class PipelineFactory:
    workflows: ClassVar[dict[str, WorkflowFunction]] = {}   # 所有已注册 workflow
    pipelines: ClassVar[dict[str, list[str]]] = {}          # 命名 pipeline

    @classmethod
    def create_pipeline(cls, config, method) -> Pipeline:
        # 优先使用 config.workflows（用户自定义），否则查找命名 pipeline
        workflows = config.workflows or cls.pipelines.get(method, [])
        return Pipeline([(name, cls.workflows[name]) for name in workflows])
```

**内置 4 种 Pipeline 方案：**

| Pipeline | 特点 |
|---|---|
| `Standard` | 全 LLM 驱动，质量最高，成本最高 |
| `Fast` | NLP 替代 LLM 抽取，成本极低，质量中等 |
| `StandardUpdate` | Standard + 增量更新 workflow |
| `FastUpdate` | Fast + 增量更新 workflow |

### 3.3 执行机制

**`run_pipeline`** 是执行引擎：

```python
async def run_pipeline(pipeline, config, callbacks, is_update_run):
    context = create_run_context(...)
    async for result in _run_pipeline(pipeline, config, context):
        yield result   # 每完成一个 workflow 即 yield，支持实时进度
```

关键设计：
- **异步迭代器**：外部调用者每完成一个 workflow 即可处理，无需等待整个 pipeline
- **状态持久化**：每个 workflow 结束后写 `stats.json` 和 `context.json`，支持断点恢复
- **增量更新**：`is_update_run=True` 时创建带时间戳的 `delta_storage`，完成后合并

### 3.4 Placement — 任务并发策略

GraphRAG 不做分布式调度，**Placement 体现在 Workflow 内部并发模型：**

```python
# derive_from_rows — 行级并发核心工具
async def derive_from_rows(
    dataframe: pd.DataFrame,
    transform: Callable,        # 对每行执行的异步函数
    callbacks: WorkflowCallbacks,
    num_threads: int,           # 并发数 = config.concurrent_requests
    async_type: AsyncType,      # AsyncIO 或 Threaded
) -> list[Any]: ...
```

- `AsyncType.AsyncIO`：`asyncio.gather` 并发协程（I/O 密集型，适合 LLM API 调用）
- `AsyncType.Threaded`：线程池并发（CPU 密集型或不支持 async 的代码）
- `num_threads = config.concurrent_requests`，全局控制 LLM API 并发数

---

## 4. 知识图谱构建流程

### Standard Pipeline 各阶段

```
原始文档
  ↓ load_input_documents       — InputReader 读取 txt/csv/pdf 等
  ↓ create_base_text_units     — Token/Sentence 分块，SHA-512 内容寻址
  ↓ create_final_documents     — 文档最终化
  ↓ extract_graph (LLM)        — 实体/关系抽取 + 多次追问（gleaning）+ 描述摘要
  ↓ finalize_graph             — 计算 degree，分配 UUID
  ↓ extract_covariates (可选)  — 从文本抽取主张/声明
  ↓ create_communities         — Leiden 层次化社区检测
  ↓ create_final_text_units    — chunk 与图元素关联
  ↓ create_community_reports   — LLM 为每个社区生成文字摘要
  ↓ generate_text_embeddings   — 实体/chunk/社区报告 → 向量存储
  ↓
知识图谱（Parquet 表 + 向量索引）
```

### 实体抽取细节（extract_graph）

```
chunk 文本 → 填入 extraction_prompt
  ↓ LLM 返回结构化文本：
  ("entity"<|>NAME<|>TYPE<|>DESCRIPTION)##("relationship"<|>SRC<|>TGT<|>DESC<|>WEIGHT)
  ↓ 若 max_gleanings > 0：继续追问 CONTINUE_PROMPT
  |  — LLM 说 "Y" → 继续追问
  |  — LLM 说 "N" 或达上限 → 停止
  ↓ 多 chunk 聚合：按 (title,type) groupby 合并实体，按 (source,target) 合并关系
  ↓ summarize_descriptions：多个描述 → 单一摘要（LLM 调用）
```

### 社区检测（Leiden 算法）

```
relationships 边表
  ↓ hierarchical_leiden(edges, max_cluster_size)
  ↓ 多层级输出：
    level 0（细粒度，具体话题）
    level 1
    level N（粗粒度，宏观主题）
  ↓ 每个社区聚合：entity_ids, relationship_ids, text_unit_ids
  ↓ LLM 生成社区报告（JSON 格式：title + summary + findings + rating）
```

### Fast Pipeline 差异

- `extract_graph_nlp`：NLP 名词短语提取替代 LLM，支持 3 种提取器（RegexEnglish / Syntactic / CFG）
- `prune_graph`：去除低频/低信息量节点和边
- **成本显著降低，但图语义精度较低**

---

## 5. 查询系统

### 5.1 四种搜索模式

| 模式 | 用途 | 主要数据来源 |
|---|---|---|
| **Local Search** | 精确实体相关问题 | entities + relationships + text_units + community_reports |
| **Global Search** | 整体主题/趋势问题 | 所有 community_reports（Map-Reduce） |
| **DRIFT Search** | 探索性多跳推理 | Global primer + Local 迭代展开 |
| **Basic Search** | 简单语义相似度 | text_units 向量搜索 |

### 5.2 Local Search 流程

```
用户查询
  ↓ query → embedding → 向量近邻搜索 → top-k 相关实体
  ↓ 展开图邻居（entity + relationship + community_reports + text_units）
  ↓ Token budget 控制拼装上下文
  ↓ 填入 LOCAL_SEARCH_SYSTEM_PROMPT → LLM 流式生成回答
```

### 5.3 Global Search 流程（Map-Reduce 架构）

```
用户查询
  ↓ 筛选指定 level 的社区报告，分批处理
  ↓ Map 阶段（asyncio.gather 高并发）：
     每批 → 填入 MAP_SYSTEM_PROMPT → LLM 返回 key points (JSON)
  ↓ Reduce 阶段（单次 LLM 调用）：
     所有 key points 按 score 降序 → 填入 REDUCE_PROMPT → 流式输出
```

### 5.4 DRIFT Search 流程

```
初始全局查询（Primer）→ 生成初始答案 + follow-up 问题列表
  ↓ 对每个 follow-up 问题：
    |— Local Search 在图谱中搜索具体信息
    |— 更新全局答案状态
    |— 生成新的 follow-up 问题（递归）
  ↓ 迭代直到 depth 限制或问题耗尽
```

### 5.5 Context Builder 抽象

```python
class GlobalContextBuilder(ABC):
    async def build_context(self, query, conversation_history, **kwargs) -> ContextBuilderResult

class LocalContextBuilder(ABC):
    def build_context(self, query, conversation_history, **kwargs) -> ContextBuilderResult

class DRIFTContextBuilder(ABC):
    async def build_context(self, query, **kwargs) -> tuple[pd.DataFrame, dict[str, int]]
```

`ContextBuilderResult` 包含：
- `context_chunks`：填入 prompt 的上下文字符串
- `context_records`：用于展示数据来源的 DataFrame 字典
- `llm_calls`, `prompt_tokens`, `output_tokens`：成本追踪

---

## 6. 存储系统

### 6.1 三类独立存储

```
input_storage    ← 原始输入文档（只读）
output_storage   ← 索引产物（Parquet 表 + JSON 状态文件）
vector_store     ← 向量嵌入（独立向量数据库）
```

### 6.2 存储后端

**表格/文件存储（graphrag-storage）：**
- `FileStorage`：本地文件系统（默认）
- `AzureBlobStorage`：Azure Blob
- `AzureCosmosStorage`：Azure Cosmos DB

**向量存储（graphrag-vectors）：**
- `LanceDBVectorStore`：本地默认
- `AzureAISearchVectorStore`：云端 Azure AI Search
- `CosmosDBVectorStore`：Azure Cosmos DB

**流式 Table 接口（核心内存优化）：**

```python
# 无需将整个表加载到内存
async for row in table:
    process(row)

await table.write(row)  # 行级写入
```

### 6.3 核心数据模型（Parquet 表结构）

| 表名 | 关键字段 |
|---|---|
| `entities` | id, title, type, description, text_unit_ids, degree |
| `relationships` | id, source, target, description, weight, text_unit_ids |
| `communities` | id, level, parent, children, entity_ids, relationship_ids |
| `community_reports` | id, level, title, summary, full_content, findings, rank |
| `text_units` | id, text, n_tokens, document_id, entity_ids, relationship_ids |
| `documents` | id, title, text, text_unit_ids, creation_date |
| `covariates`（可选） | id, covariate_type, subject_id, object_id, status, start_date, end_date |

---

## 7. LLM 抽象层（graphrag-llm）

### 7.1 核心抽象

```python
class LLMCompletion(ABC):
    def completion(/, **kwargs) -> LLMCompletionResponse | Iterator[LLMCompletionChunk]
    async def completion_async(/, **kwargs) -> LLMCompletionResponse | AsyncIterator[LLMCompletionChunk]
    def completion_thread_pool(...) -> Iterator[ThreadedLLMCompletionFunction]  # 线程池批处理
    def completion_batch(...) -> list[...]                                        # 批量处理

class LLMEmbedding(ABC):
    def embed(texts: list[str]) -> list[list[float]]
    async def embed_async(texts: list[str]) -> list[list[float]]
```

### 7.2 工厂与延迟注册

所有服务（LLM/Storage/Vector）基于 `graphrag-common` 的泛型 `Factory[T]`：
- `scope="singleton"`：基于参数 hash 缓存实例
- `scope="transient"`：每次创建新实例
- **延迟注册（Lazy Registration）**：首次请求时才 import 实现类，避免未使用的重依赖提前加载

**内置实现：** `LiteLLMCompletion / LiteLLMEmbedding`，基于 LiteLLM 库，支持 100+ LLM Provider（OpenAI、Azure、Anthropic、Ollama 等）。

---

## 8. 配置系统

### 8.1 顶级配置结构（GraphRagConfig）

```python
class GraphRagConfig(BaseModel):
    completion_models: dict[str, ModelConfig]   # 命名的 completion 模型字典
    embedding_models: dict[str, ModelConfig]    # 命名的 embedding 模型字典
    concurrent_requests: int                     # 全局并发数
    async_mode: AsyncType                        # AsyncIO 或 Threaded

    input_storage: StorageConfig
    output_storage: StorageConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    workflows: list[str] | None                  # 自定义 workflow 列表

    extract_graph: ExtractGraphConfig
    cluster_graph: ClusterGraphConfig
    community_reports: CommunityReportsConfig
    embed_text: EmbedTextConfig

    local_search: LocalSearchConfig
    global_search: GlobalSearchConfig
    drift_search: DRIFTSearchConfig
    basic_search: BasicSearchConfig
```

### 8.2 命名模型引用机制

```yaml
completion_models:
  default:
    model: gpt-4o
    api_key: ${GRAPHRAG_API_KEY}
  cheap:
    model: gpt-4o-mini
    api_key: ${GRAPHRAG_API_KEY}

extract_graph:
  completion_model_id: default    # 引用命名模型
summarize_descriptions:
  completion_model_id: cheap      # 不同阶段用不同规格模型
```

不同阶段使用不同规格模型，统一在顶层管理 API key。

---

## 9. Callbacks 系统

### 9.1 WorkflowCallbacks（索引阶段）

```python
class WorkflowCallbacks(Protocol):
    def pipeline_start(self, names: list[str]) -> None
    def pipeline_end(self, results: list[PipelineRunResult]) -> None
    def workflow_start(self, name: str, instance: object) -> None
    def workflow_end(self, name: str, instance: object) -> None
    def progress(self, progress: Progress) -> None
    def pipeline_error(self, error: BaseException) -> None
```

### 9.2 QueryCallbacks（查询阶段）

```python
class QueryCallbacks(Protocol):
    def on_context(self, context: Any) -> None
    def on_map_response_start(self, map_responses) -> None
    def on_map_response_end(self, map_responses) -> None
    def on_llm_new_token(self, token: str) -> None  # SSE 流式输出接入点
```

`on_llm_new_token` 是实现 Server-Sent Events 流式输出的关键扩展点。

---

## 10. Prompt Tune 系统

自动为特定语料生成最优 Prompt：

| 生成器 | 功能 |
|---|---|
| `domain.py` | 自动推断文档领域（如"医学研究"、"法律"） |
| `persona.py` | 为 LLM 生成领域专家角色描述 |
| `entity_types.py` | 自动推断值得关注的实体类型 |
| `entity_relationship.py` | 生成实体-关系抽取 Prompt 示例 |
| `community_report_rating.py` | 生成社区报告评分标准 |
| `language.py` | 检测文档主要语言 |

**使用流程：**
1. 采样一批文档（随机/top/所有模式）
2. LLM 分析样本，推断领域、实体类型、语言
3. 生成针对该语料优化的 Prompt 模板
4. 将生成的 Prompt 保存到项目目录

---

## 11. 扩展性设计

### 自定义 Workflow

```python
from graphrag.index.workflows.factory import PipelineFactory

async def my_workflow(config, context):
    ...

PipelineFactory.register("my_workflow", my_workflow)
```

然后在 `settings.yaml` 的 `workflows:` 列表中插入 `my_workflow`。

### 自定义 LLM Provider

```python
from graphrag_llm.completion import register_completion

class MyLLM(LLMCompletion):
    ...

register_completion("my_llm", MyLLM)
# 配置中指定 type: my_llm
```

### 公开 Python API

```python
import graphrag.api as api

# 索引
async for result in api.build_index(config, method, callbacks):
    print(result)

# 查询
response, context = await api.local_search(config, entities, communities, ...)
response, context = await api.global_search(config, entities, communities, ...)

# 流式查询
async for chunk in api.local_search_streaming(config, ...):
    print(chunk, end="")
```

---

## 12. 关键技术决策总结

| 决策 | 说明 |
|---|---|
| **流式处理** | 所有 workflow 改为行级异步流式读写，避免大数据集 OOM |
| **SHA-512 内容寻址** | Text unit ID 基于内容 hash，保证幂等性和跨运行去重 |
| **Gleaning 机制** | 实体抽取后迭代追问 LLM，显著提升实体召回率 |
| **层次化 Leiden 社区** | 多粒度社区树，支持宏观/微观两种查询视角 |
| **Map-Reduce 全局搜索** | 社区报告并行处理，将 N 个报告处理从串行降至并行 |
| **Protocol 接口** | 大量使用 Python Protocol 替代抽象基类，降低耦合 |
| **缓存驱动成本控制** | 基于内容 hash 的 LLM 响应缓存，大幅降低重复索引成本 |
| **命名模型引用** | 不同阶段可配置不同规格模型，统一管理 API key |
| **延迟注册** | 工厂按需 import 实现类，避免未使用依赖提前加载 |

---

## 13. 关键文件路径索引

| 功能 | 文件 |
|---|---|
| Pipeline 工厂 | `packages/graphrag/graphrag/index/workflows/factory.py` |
| Pipeline 执行引擎 | `packages/graphrag/graphrag/index/run/run_pipeline.py` |
| Workflow 类型定义 | `packages/graphrag/graphrag/index/typing/workflow.py` |
| Pipeline 上下文 | `packages/graphrag/graphrag/index/typing/context.py` |
| 全局配置模型 | `packages/graphrag/graphrag/config/models/graph_rag_config.py` |
| 实体图抽取（LLM） | `packages/graphrag/graphrag/index/operations/extract_graph/graph_extractor.py` |
| 实体图抽取（NLP） | `packages/graphrag/graphrag/index/workflows/extract_graph_nlp.py` |
| 社区检测 | `packages/graphrag/graphrag/index/operations/cluster_graph.py` |
| 社区报告生成 | `packages/graphrag/graphrag/index/operations/summarize_communities/summarize_communities.py` |
| Local Search | `packages/graphrag/graphrag/query/structured_search/local_search/search.py` |
| Global Search | `packages/graphrag/graphrag/query/structured_search/global_search/search.py` |
| DRIFT Search | `packages/graphrag/graphrag/query/structured_search/drift_search/search.py` |
| LLM 抽象基类 | `packages/graphrag-llm/graphrag_llm/completion/completion.py` |
| LLM 工厂 | `packages/graphrag-llm/graphrag_llm/completion/completion_factory.py` |
| 存储工厂 | `packages/graphrag-storage/graphrag_storage/storage_factory.py` |
| 向量存储工厂 | `packages/graphrag-vectors/graphrag_vectors/vector_store_factory.py` |
| 通用工厂基类 | `packages/graphrag-common/graphrag_common/factory/factory.py` |
| 数据模型字段 | `packages/graphrag/graphrag/data_model/schemas.py` |
| 公开 API 入口 | `packages/graphrag/graphrag/api/__init__.py` |
| CLI 入口 | `packages/graphrag/graphrag/cli/main.py` |
| Workflow 回调协议 | `packages/graphrag/graphrag/callbacks/workflow_callbacks.py` |
| 查询回调协议 | `packages/graphrag/graphrag/callbacks/query_callbacks.py` |
| 行级并发工具 | `packages/graphrag/graphrag/index/utils/derive_from_rows.py` |


 Vector RAG 三组件 vs GraphRAG 对应关系                                                                                                      
                                                                                                                                              
  总览                                                                                                                                        
                                                                                                                                              
  Vector RAG Pipeline          GraphRAG 对应                
  ─────────────────────        ──────────────────────────────────────
  Query Rewriter          →    【不存在专用组件】部分由对话历史拼接替代
         ↓
  Retrieval               →    【两阶段异构检索】
    ├─ Vector Search      →      map_query_to_entities (entity embedding 向量检索)
    └─ (chunk fetch)      →      图结构展开 (relationship/community/text_unit 联动)
         ↓
  Reranker                →    【不存在 cross-encoder 重排】图结构 rank 指标替代
         ↓
  LLM Generation          →    LocalSearch / GlobalSearch 的 completion_async

  ---
  1. Query Rewriter → 无专用组件

  GraphRAG 的唯一近似处理在 mixed_context.py:131-135：

  if conversation_history:
      pre_user_questions = "\n".join(
          conversation_history.get_user_turns(conversation_history_max_turns)
      )
      query = f"{query}\n{pre_user_questions}"  # 把历史轮次拼到 query

  这只是对话历史拼接，不是真正的 query rewriting。没有 query expansion、sub-query decomposition、HyDE 等机制。

  DRIFT Search 的 Primer 阶段可以视为"全局 query decomposition"，但它是检索-生成的一部分，不是独立的 rewriter。

  ---
  2. Retrieval → 两阶段异构检索

  这是 GraphRAG 与 vector RAG 差异最大的部分，检索分两个完全不同的阶段：

  Stage 1：实体向量检索（entity_extraction.py:62）

  # 唯一的向量检索入口
  text_embedding_vectorstore.similarity_search_by_text(
      text=query,
      text_embedder=lambda t: text_embedder.embedding(input=[t]).first_embedding,
      k=top_k_mapped_entities * oversample_scaler,  # oversample=2x 过采样
  )
  # 返回：top-k Entity 对象列表

  检索的是实体（不是 chunk），用的是 entity_description_embedding 向量索引。

  Stage 2：图结构联动展开（mixed_context.py:172-215，纯 CPU）

  selected_entities
    ├─ _build_community_context
    │    entity.community_ids → community_reports (按 matches_count + rank 排序)
    ├─ _build_local_context
    │    entity.title → relationships (in-network 优先，再按 links+rank 排序)
    └─ _build_text_unit_context
         entity.text_unit_ids → text_units (按 entity_order + num_relationships 排序)

  Stage 2 完全不调用向量检索，是图的邻居展开 + pandas 过滤，没有 LLM 参与。

  ---
  3. Reranker → 无 cross-encoder，图结构 rank 指标替代

  GraphRAG 中排序由四套独立的图结构指标完成，每层单独排序：

  ┌────────┬──────────────────────────────────────────────┬──────────────────────────┐
  │   层   │                    排序键                    │         代码位置         │
  ├────────┼──────────────────────────────────────────────┼──────────────────────────┤
  │ 实体层 │ 向量相似度（ANN 返回顺序）                   │ entity_extraction.py:62  │
  ├────────┼──────────────────────────────────────────────┼──────────────────────────┤
  │ 社区层 │ (matched_entity_count, community.rank)       │ mixed_context.py:256-258 │
  ├────────┼──────────────────────────────────────────────┼──────────────────────────┤
  │ 关系层 │ in-network 优先 → (links_count, rank/weight) │ local_context.py:240-314 │
  ├────────┼──────────────────────────────────────────────┼──────────────────────────┤
  │ 文本层 │ (entity_order, -relationship_count)          │ mixed_context.py:339     │
  └────────┴──────────────────────────────────────────────┴──────────────────────────┘

  最终用 token budget 截断（local_context.py:77）作为硬性 cutoff：

  if current_tokens + new_tokens > max_context_tokens:
      break   # 按排序顺序塞满 context window 即截止

  这与 vector RAG 的 cross-encoder reranker（语义相关性重打分）性质完全不同——GraphRAG 的"排序"是基于图中心度/结构重要性，而非 query 相关性。

 就是 Local/Global Search 里最后那一次 LLM 生成回答的调用。                                                                                  
                                                                                                                                              
  看 local_search/search.py:100-113：
                                                                                                                                              
  # self.model 是 LLMCompletion 实例（LiteLLMCompletion）                                                                                   
  response = await self.model.completion_async(                                                                                               
      messages=[                                                                                                                              
          {"role": "system", "content": search_prompt},  # 组装好的上下文
          {"role": "user",   "content": query},           # 用户原始问题
      ],
      stream=True,
      **self.model_params,
  )

  async for chunk in response:
      response_text = chunk.choices[0].delta.content or ""
      full_response += response_text

  completion_async 是 LLMCompletion 抽象基类定义的异步方法，LiteLLMCompletion 实现它，底层调用 litellm.acompletion()，再由 LiteLLM
  路由到实际的 LLM（OpenAI API / 本地 Ollama / 任意 provider）。
  ---
  本地 LLM 时各组件的 placement

  completion_models:
    local_llm:           # 主力（社区报告生成、查询生成）
      model_provider: ollama
      model: llama3.1
      api_base: http://localhost:11434
      api_key: "ollama"

    local_llm_fast:      # 轻量（如果实现 query rewriter）
      model_provider: ollama
      model: qwen2.5:3b
      api_base: http://localhost:11434
      api_key: "ollama"

  embedding_models:
    local_embed:         # Stage 1 向量检索专用
      model_provider: ollama
      model: nomic-embed-text
      api_base: http://localhost:11434
      api_key: "ollama"

  ┌────────────────────────┬─────────────────────────────┬───────────────────────────────────────┬────────────────────────────────────────┐
  │          组件          │          对应资源           │               配置字段                │                  说明                  │
  ├────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────┤
  │                        │ （不存在）自定义实现用轻量  │                                       │ 用 local_llm_fast 生成扩展             │
  │ Query Rewriter         │ LLM                         │ 无内置字段，需包装 search() 调用      │ query，替换传入 search(query=...)      │
  │                        │                             │                                       │ 的参数                                 │
  ├────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────┤
  │ Stage 1 向量检索       │ Embedding 模型 + LanceDB    │ local_search.embedding_model_id:      │ Embedding 走 Ollama GPU，LanceDB ANN   │
  │                        │                             │ local_embed                           │ 搜索 CPU                               │
  ├────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────┤
  │ Stage 2 图展开         │ 纯 CPU pandas               │ 无 LLM 配置                           │ 不消耗任何模型资源                     │
  ├────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────┤
  │ "Reranker"（结构排序） │ 纯 CPU                      │ 无 LLM 配置                           │ degree/rank 计算，不可替换为语义       │
  │                        │                             │                                       │ reranker（需子类化 ContextBuilder）    │
  ├────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼────────────────────────────────────────┤
  │ LLM 生成               │ 主力 LLM                    │ local_search.completion_model_id:     │ VRAM bound                             │
  │                        │                             │ local_llm                             │                                        │
  └────────────────────────┴─────────────────────────────┴───────────────────────────────────────┴────────────────────────────────────────┘
