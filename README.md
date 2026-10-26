# 优选商城 · RAG 智能客服 Copilot

一个面向电商客服场景的 **RAG + 订单工具 + 人工兜底** 演示项目。

- 设计文档：`技术设计方案_RAG智能客服_v1.md`
- 业务原型：`02_智能客服.md`
- 当前形态：**优先级分流 + Copilot**（P3 低优先自动回复；P1/P0 兜底并转人工）
- 生成模型：**智谱 GLM-4.7-Flash**（公有 API，关闭思考模式）
- 向量模型：**硅基流动 BAAI/bge-m3**（1024 维，Sentence Transformers 兼容模型）
- 重排模型：**硅基流动 BAAI/bge-reranker-v2-m3**（cross-encoder）
- 向量库：**ChromaDB**
- 技术栈：Python 3.12 + FastAPI + SQLite + ChromaDB + SiliconFlow

---

## 1. 已实现的能力

| 能力 | 说明 |
|---|---|
| 四通道分流 | FAQ / RAG / 订单物流 Tool / 人工兜底 |
| 优先级分流 | P0–P3 四级；规则兜底 + GLM 分级，模型只能抬高、不能降低规则优先级 |
| 低优先自动回复 | P3 低风险问题由 GLM 决定 `auto_reply`；只有模型允许且置信度达标才直接回复 |
| 自动接待 | 收到问题先自动发送“已收到，正在查询”的接待语，再由 GLM 审核并决定后续 |
| 类目兜底 | 商品质量/资金差价/大额订单/物流/发票/地址/投诉分别配置话术与技能组 |
| 高风险兜底 | 退款/差价/商品质量/大额订单/投诉 → 类目兜底话术 + 转人工 |
| 合规话术 | 委婉表达、不承诺一定退款/赔偿、说明“具体以平台规则和审核结果为准” |
| 文本切块 | 按 `\n\n` 切分，不限制字数；种子知识库 12 文档 → 36 块 |
| FAQ 秒回 | 精确匹配 + 字符 n-gram 语义匹配 + 阈值/分差双重门槛 |
| 向量检索 | ChromaDB + SiliconFlow `BAAI/bge-m3`，召回 15 条 |
| 模型重排 | SiliconFlow `BAAI/bge-reranker-v2-m3` cross-encoder，重排到 5 条 |
| 检索降级 | Chroma/SiliconFlow 不可用时，自动回退 BM25 + 字符 n-gram 词法检索 |
| 订单/物流查询 | 只读 Mock API：订单状态、物流、退款进度 |
| 上下文改写 | 多轮追问时补全“这个/那/定金”等指代 |
| 风险预检 | 退款、退差价、投诉、改地址、取消订单等涉敏动作转人工 |
| 引用来源 | 每条建议展示知识库标题、条款、分数、摘要 |
| 决策追踪 | 展示规范化、风控、路由、优先级、FAQ、召回、重排、Tool、生成、输出风控 |
| 转人工上下文包 | 摘要、订单快照、引用、草稿、风险标签、技能组、优先级 |
| 反馈闭环 | 采用 / 编辑 / 有用 / 无用 / 答错 / 政策过期 |
| 降级 | 模型限流/失败自动降级为本地模板，并在界面标记 |
| 评测 | 28 条 pytest 覆盖四通道、优先级、自动回复、类目兜底、接待语、合规话术、切块、多轮、反馈、转人工 |

---

## 2. 快速启动

### 2.1 安装依赖

项目有两套环境：

**① 项目基础环境（测试、词法回退）**

```powershell
cd D:\AI客服
uv sync
```

**② 向量环境 `D:\pytorch\.venv`（ChromaDB + SiliconFlow）**

```powershell
uv venv --python 3.12 D:\pytorch\.venv

uv pip install --python D:\pytorch\.venv\Scripts\python.exe `
  fastapi "uvicorn[standard]" pydantic pydantic-settings httpx sqlalchemy `
  python-dotenv chromadb sentence-transformers pytest
```

`run_demo.bat` 会优先使用 `D:\pytorch\.venv`；如果找不到，会回退到项目 `.venv` 的词法检索模式。

### 2.2 配置模型

复制 `.env.example` 为 `.env`：

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` 关键配置：

```dotenv
ZHIPU_API_KEY=你的智谱APIKey
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL=glm-4.7-flash
ZHIPU_DISABLE_THINKING=true
ZHIPU_MAX_TOKENS=800
ZHIPU_TIMEOUT_SECONDS=10
ZHIPU_TOTAL_TIMEOUT_SECONDS=8
ZHIPU_MAX_RETRIES=2
ZHIPU_RETRY_BACKOFF_SECONDS=1
ZHIPU_COOLDOWN_SECONDS=20
PRIORITY_TOTAL_TIMEOUT_SECONDS=3
PRIORITY_LARGE_AMOUNT_THRESHOLD=500
LLM_MODE=zhipu

SILICONFLOW_API_KEY=你的硅基流动APIKey
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-m3
SILICONFLOW_RERANK_MODEL=BAAI/bge-reranker-v2-m3
EMBEDDING_PROVIDER=siliconflow
RERANK_PROVIDER=siliconflow

RETRIEVAL_BACKEND=chroma
CHROMA_PATH=./data/chroma
CHROMA_COLLECTION=kb_chunks
RECALL_TOP_K=15
RERANK_TOP_K=5
CONTEXT_TOP_K=5
RERANK_MIN_SCORE=0.10
```

说明：

- `EMBEDDING_PROVIDER=siliconflow`：使用硅基流动 `BAAI/bge-m3`，1024 维，不占用本地显存。
- `RERANK_PROVIDER=siliconflow`：使用硅基流动 `BAAI/bge-reranker-v2-m3` cross-encoder。
- `RETRIEVAL_BACKEND=chroma`：使用 ChromaDB；`auto` 会自动尝试 Chroma，失败回退词法；`lexical` 强制 BM25。
- `RECALL_TOP_K=15` / `RERANK_TOP_K=5`：召回 15 条，模型重排后保留 5 条。
- `SILICONFLOW_API_KEY` 只放在 `.env`，不会进入版本库。
- 本地 Sentence Transformers 可选：把 `EMBEDDING_PROVIDER` / `RERANK_PROVIDER` 改成 `local`；当前 `transformers 5.20.2` + BGE 小模型存在兼容问题，默认走 SiliconFlow。
- 智谱限流时仍会自动降级为本地模板，不影响主链路。

### 2.3 启动

推荐双击：

```powershell
.\run_demo.bat
```

或手动用向量环境启动：

```powershell
cd D:\AI客服
D:\pytorch\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

健康检查会显示当前检索后端：

```text
http://127.0.0.1:8000/api/v1/health
```

或双击/运行：

```powershell
.\run_demo.bat
```

打开浏览器访问：

```text
http://127.0.0.1:8000
```

---

## 3. Demo 场景

页面左侧有 8 个快捷场景，建议按顺序体验：

| 场景 | 顾客问题 | 通道 | 优先级/策略 | 演示点 |
|---|---|---|---|---|
| 订单还没发货 | 我的订单还没发货怎么办 | Tool | P2 · Copilot | 查订单 + 引用发货时效规则 |
| 怎么退货 | 怎么退货 | FAQ | P2 · Copilot | 高频标准问毫秒级命中 |
| 预售定金能退吗 | 预售商品定金能退吗 | RAG | P1 · Copilot | 长尾政策检索 + 人工复核建议 |
| 退款到账了吗 | 退款到账了吗 | Tool | P2 · Copilot | 只读查退款进度，不执行写操作 |
| 我要退差价 | 我要退差价 | Human | P1 · 兜底 + 转人工 | 涉钱写操作，兜底话术 + 上下文包 |
| 多轮追问 | 先问“预售什么时候发货”，再问“那定金能退吗” | RAG | P1 · Copilot | 上下文改写 + 指代补全 |
| 大额商品故障 | 我买的扫地机器人用了两天就坏了，要退款 | Human | P1 · 兜底 + 转人工 | 大额订单 ¥1299 + 商品质量，强制兜底 |
| 低优先自动回复 | 运费怎么算 | FAQ | P3 · 自动回复 | 低风险规则问，模型自动回复 |

每个场景会重置会话，保证演示结果稳定。右上角可切换“低优先自动回复”开关：开启时 P3 自动发送，关闭时所有问题都回到 Copilot 建议模式。

> 交互顺序：顾客提问 → 系统立即自动发送接待语“您好，已收到您的问题，我先帮您查询一下，请稍候。” → GLM 做优先级审核 → 根据结果自动回复 / Copilot / 兜底转人工。

### Demo 预览

低优先自动回复：

![低优先自动回复](docs/demo-priority-auto.png)

大额商品质量问题的兜底与转人工：

![大额商品兜底](docs/demo-priority-fallback.png)

“差价”类资金问题的兜底与转人工：

![差价兜底](docs/demo-priority-money.png)

---

## 4. 架构

```text
顾客 / 客服工作台
        │
        ▼
   Chat API（FastAPI）
        │
        ▼
   预处理 & 风控
   ├─ 文本规范化 / 乱码修复
   ├─ 敏感词 / 涉敏动作识别
   └─ 会话上下文 / 指代判断
        │
        ▼
   优先级判定（P0–P3）
   ├─ 规则兜底：安全/资金/商品/大额/投诉 → 抬高优先级
   ├─ GLM 分级：P2/P3 模糊问题做补充判断
   └─ 模型只能抬高，不能降低规则优先级
        │
        ▼
      意图路由
   ├─ P3 低优先：FAQ/RAG 结果 → 自动回复（可开关）
   ├─ P2 中优先：FAQ/RAG/Tool → Copilot 建议 → 人工复核
   ├─ P1 高优先：兜底话术 + 转人工（商品/大额/退款/投诉）
   └─ P0 紧急：直接转人工 + 上下文包
        │
        ▼
   RAG 通道（知识问）
   ├─ 切块：按 \n\n 切分，不限制字数
   ├─ 向量化：SiliconFlow BAAI/bge-m3（1024 维）
   ├─ 向量库：ChromaDB（cosine）
   ├─ 召回：15 条
   ├─ 重排：SiliconFlow BAAI/bge-reranker-v2-m3 → 5 条
   └─ 生成：Top 5 + 订单快照 + 历史 → GLM
        │
        ▼
   输出风控 → 建议话术 / 引用 / 风险提示
        │
        ▼
   Copilot 工作台（自动回复 / 采用 / 编辑 / 换一个 / 转人工）
```

### 4.1 RAG 检索链路

| 环节 | 当前实现 |
|---|---|
| 文本切块 | 按 `\n\n` 切分，不限制字数；种子知识库 12 文档 → 36 块 |
| Embedding | SiliconFlow `BAAI/bge-m3`，1024 维 |
| 向量库 | ChromaDB，cosine 距离，持久化到 `data/chroma` |
| 召回 | `RECALL_TOP_K=15` |
| Rerank | SiliconFlow `BAAI/bge-reranker-v2-m3` cross-encoder，保留 5 条 |
| 生成上下文 | 重排 Top 5 + 订单快照 + 最近会话 |
| 生成 | 智谱 GLM-4.7-Flash；失败/限流时用抽取式模板 |
| 降级 | Chroma/SiliconFlow 不可用 → BM25 + 字符 n-gram 词法检索 |
| 引用 | 返回重排后的标题、章节、分数、摘要 |

> 首次启动会用 SiliconFlow 对 36 个块做一次 Embedding 并写入 Chroma；之后通过 manifest 复用，不会重复向量化。

### 4.2 优先级与兜底策略

| 优先级 | 典型问题 | 默认处理 | 是否自动回复 |
|---|---|---|---|
| **P0 紧急** | 人身安全、食品安全、诈骗/盗刷、账户被盗、法律/监管、媒体曝光 | 立即转人工，优先队列 | 否 |
| **P1 高优先** | 退款/退差价/退定金/赔偿、商品质量/破损/错漏发、大额订单售后、投诉 | 兜底话术 + 转人工；纯规则问可 Copilot | 否 |
| **P2 中优先** | 物流异常、退货换货流程、退款进度、发票、改地址、订单状态 | Copilot 建议，人工复核 | 否 |
| **P3 低优先** | 运费/包邮、发货时效、会员权益、一般流程咨询、问候 | 模型自动回复（可开关） | 是 |

**关键安全规则**：

1. **规则是下限，模型是补充**：规则命中 P0/P1、商品问题、大额订单时，不再让模型降低优先级。
2. **商品问题必须兜底**：质量、破损、故障、错漏发、假货、过期、安全问题 → 至少 P1，使用“保留凭证 + 转售后专席”的兜底话术。
3. **大额订单必须兜底**：订单金额 ≥ `PRIORITY_LARGE_AMOUNT_THRESHOLD`（默认 500）且涉及售后/退款/商品/投诉 → 至少 P1。
4. **涉钱写操作必须兜底**：我要退款/退差价/退定金/赔偿 → P1 兜底 + 转人工；仅查询“退款到账了吗”属于 P2 状态查询。
5. **“差价”一律转人工兜底**：`差价`、`退差价`、`价保` 等资金词命中即 P1，不自动回复，具体以平台规则和审核结果为准。
6. **特殊关键词直接兜底**：`商品坏了`、`差价`、`质量有问题`、`破损`、`故障`、`漏发`、`错发`、`假货`、`过期`、`变质`、`不安全` 等命中即至少 P1，不再看金额。
7. **低优先自动回复有门槛**：仅 P3、无商品/资金/大额/投诉风险、置信度 ≥ 0.6 时自动发送；否则进入 Copilot。
8. **所有 P1/P0 都保留人工入口**：即使模型给出兜底话术，也不自动执行退款、改地址、取消订单等写操作。

**P3 / P1 / P2 的大模型决策**：

- P3：先由规则判断候选，再由 GLM 返回 `priority + auto_reply + need_human`。低风险且 `auto_reply=true` 时，RAG 检索结果会作为上下文交给 GLM 生成答案并直接返回顾客。
- P1/P2：对于非强制兜底的纯规则问题，由 GLM 根据问题类型返回 `need_human`，决定“Copilot 建议”还是“兜底 + 转人工”。
- 强制兜底关键词、商品问题、大额订单、资金写操作、投诉：不调用模型做降级判断，直接走规则兜底。
- 模型超时/限流时：自动回退规则分级，主链路不会卡死。

**自动接待与审核时序**：

1. 顾客提问。
2. 系统立即自动发送接待语：“您好，已收到您的问题，我先帮您查询一下，请稍候。”（模板，非模型生成）
3. GLM 审核：优先级、类目、`auto_reply`、`need_human`。
4. 根据审核结果：
   - P3 + `auto_reply=true` → RAG/FAQ → GLM 生成 → 直接返回顾客；
   - P2 / P1 纯规则 → Copilot 建议，人工复核后发送；
   - P1 强制兜底 / P0 → 类目兜底话术 + 转人工。

**合规/法务措辞（参考常见电商表达）**：

- 不承诺“一定退款/一定赔偿/一定退差价”，不使用“肯定能退/不能退”等绝对结论。
- 使用“建议您保留商品、包装及相关凭证”“具体以平台规则和审核结果为准”“我们会尽快为您核实”“给您带来不便非常抱歉”等委婉表达。
- 商品问题：先致歉 + 保留凭证 + 转售后核实。
- 资金/差价：说明需结合订单、活动规则和支付信息核实，不直接承诺退差。
- 物流：转物流专席核实节点；破损/丢失建议保留物流凭证。
- 发票：转财务专席，具体以税务和平台规则为准。

**类目兜底话术**：

| 类目 | 触发示例 | 兜底话术方向 | 技能组 |
|---|---|---|---|
| 安全/法律 | 过敏、诈骗、盗刷、起诉 | 优先转人工，请勿额外操作 | 投诉专席 |
| 商品质量 | 商品坏了、质量有问题、破损、故障 | 保留商品/包装/凭证，转售后专席 | 售后专席 |
| 资金/差价 | 差价、退差价、退定金、赔偿 | 资金操作需人工核实，转售后专席 | 售后专席 |
| 大额订单 | 金额 ≥ 500 且涉及售后 | 大额订单需人工核实 | 售后专席 |
| 物流 | 物流异常、丢件、签收未收到 | 转物流专席核实节点 | 物流专席 |
| 发票 | 开票、红冲、发票抬头 | 转财务专席核实 | 财务专席 |
| 地址/订单 | 修改地址、取消订单 | 转订单服务专席核实履约状态 | 订单服务专席 |
| 投诉/舆情 | 投诉、曝光、差评 | 转投诉专席优先跟进 | 投诉专席 |

### 目录结构

```text
app/
  api/routes.py              # Chat / Feedback / Handoff / Demo API
  config.py                  # .env 配置
  db.py                      # SQLite 审计、反馈、转人工记录
  main.py                    # FastAPI 应用与静态资源
  schemas.py                 # Pydantic 请求/响应模型
  core/text.py               # 规范化、分词、相似度、PII 脱敏
  data/seed.py               # FAQ、知识库、订单、场景种子数据
  data/orders.py             # 只读订单/物流/退款 Mock 工具
  orchestration/router.py    # 风险预检、意图识别、路由
  orchestration/priority.py  # P0–P3 规则 + 大模型分级 + 兜底策略
  orchestration/session.py   # 会话状态（Demo 用内存）
  orchestration/orchestrator.py  # 主链路编排、Trace、Handoff
  rag/faq.py                 # FAQ 匹配
  rag/chunking.py            # 按 \n\n 切块，不限制字数
  rag/chroma_store.py        # ChromaDB 持久化、召回、词法回退
  rag/embeddings.py          # SiliconFlow BGE-M3 / 本地 Sentence Transformers
  rag/rerank.py              # SiliconFlow / 本地 BGE reranker
  rag/factory.py             # 按配置组装 Retriever + Reranker
  rag/index.py               # BM25 + 字符 n-gram 词法回退
  rag/generator.py           # GLM-4.7-Flash 调用 + 本地降级模板
  static/                    # Copilot 工作台前端
tests/test_pipeline.py       # 24 条端到端测试
tests/test_chunking.py       # 4 条切块测试
scripts/                     # 模型连通性、检索、Live 链路调试脚本
```

---

## 5. API

### `POST /api/v1/chat`

```json
{
  "session_id": "s-2231",
  "message": "我的订单还没发货怎么办",
  "user_id": "c99001",
  "mode": "copilot",
  "channel": "workbench",
  "context": {
    "order_id": "20261025001",
    "locale": "zh-CN"
  },
  "regenerate": false
}
```

响应包含：

- `reply`：建议话术 / 兜底话术 / 自动回复内容
- `channel`：`faq` / `rag` / `tool` / `human`
- `priority`：`level`（P0–P3）、`category` / `category_label`（类目）、`handling`（自动回复/Copilot/兜底/人工）、`reason`、`source`（rule / rule+llm）、`auto_reply_eligible`
- `auto_replied`：本次是否已在 `self_service` 模式下自动回复
- `acknowledgement`：自动接待语（新消息返回，重新生成时为 null）
- `confidence`、`need_human`
- `citations`：引用来源
- `suggested_actions`：建议动作
- `risk_flags`：风险标记（含商品质量、大额订单、资金风险等）
- `order_snapshot`：订单快照
- `trace`：决策链路（含自动接待、优先级审核、Chroma 召回 15、BGE 重排 5）
- `handoff`：转人工上下文包
- `debug`：FAQ 分数、检索命中、Tool 结果、优先级、模型模式

> `mode` 支持 `copilot` 与 `self_service`。`self_service` 下只有 P3 低优先问题会自动回复；P1/P0 仍然兜底并等待人工确认。

### 其他接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查、模型模式、检索后端、Embedding/Reranker、召回数量 |
| GET | `/api/v1/demo/scenarios` | Demo 场景 |
| GET | `/api/v1/demo/orders` | Mock 订单列表 |
| POST | `/api/v1/demo/seed` | 预置多轮会话 |
| POST | `/api/v1/demo/reset` | 重置 Demo 会话 |
| POST | `/api/v1/feedback` | 采用/编辑/有用/无用/答错/政策过期 |
| POST | `/api/v1/handoff` | 创建转人工工单 |

`/api/v1/health` 会返回类似：

```json
{
  "retrieval_backend": "ChromaDB + SiliconFlow/BAAI/bge-m3",
  "embedding_model": "SiliconFlow/BAAI/bge-m3",
  "reranker": "SiliconFlow/BAAI/bge-reranker-v2-m3",
  "rerank_active": true,
  "recall_top_k": 15,
  "rerank_top_k": 5
}
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

---

## 6. 测试

```powershell
uv run pytest
```

覆盖：

- FAQ 命中
- RAG 长尾检索与引用
- 按 `\n\n` 切块，不限制字数
- 种子文档 12 篇 → 36 块
- 订单 Tool 查询
- 退款进度 Tool 查询
- 退差价转人工 + 上下文包
- 多轮上下文改写
- 越界问题兜底
- 反馈接口
- 转人工接口
- 首页与健康检查
- P3 低优先自动回复
- 自动接待语先发、再审核
- 大额 + 商品质量问题兜底
- 资金写操作兜底
- “差价”关键词兜底（含“差价怎么算”）
- “商品坏了”关键词兜底
- 类目话术与技能组匹配
- 合规/委婉兜底措辞
- 规则优先级不能被大模型降低
- 大模型可以把 P3 抬高到 P1
- 大模型可以为 P1 纯规则问题决定 need_human
- 大模型可以拒绝 P3 自动回复

> 测试使用项目 `.venv` + `RETRIEVAL_BACKEND=lexical`，不依赖 ChromaDB/SiliconFlow，保证离线可跑。真实 Chroma + SiliconFlow 链路用 `scripts/test_chroma_live.py` 验证。

---

## 7. 当前实现与生产方案的差异

| 能力 | Demo | 生产建议 |
|---|---|---|
| 优先级分流 | 关键词规则 + GLM 分级（P2/P3） | 可配置规则引擎 + 微调分类模型 + 人工抽检 |
| 低优先自动回复 | P3 且置信度 ≥ 0.6 自动回复 | 按类目灰度、按渠道灰度、可一键熔断 |
| 高风险兜底 | 模板兜底 + 转人工 | 资金/商品/大额专用兜底话术 + 工单联动 |
| 文本切块 | 按 `\n\n` 切分，不限制字数 | 结构化父子块 + 表格/条款/生效期处理 |
| 向量检索 | ChromaDB + SiliconFlow BGE-M3，召回 15 | Milvus 2.5 + BGE-M3 dense/sparse + RRF |
| 重排 | SiliconFlow BGE-reranker-v2-m3，保留 5 | 自托管 BGE-reranker-v2-m3 + 业务特征微调 |
| 检索降级 | Chroma/SiliconFlow 失败 → BM25 + 字符 n-gram | 多级降级 + 熔断 + 缓存 |
| 数据库 | SQLite | MySQL 8.0 |
| 会话 | 内存 | Redis + MySQL 持久化 |
| FAQ 缓存 | 内存/无 | Redis 精确缓存 + 版本失效 |
| 文档入库 | 代码内种子数据，按 `\n\n` 重建切块 | MinerU/PyMuPDF/PaddleOCR + 版本/生效期/蓝绿索引 |
| 订单接口 | Mock 函数 | 真实只读 OpenAPI，超时/熔断/最小权限 |
| 鉴权 | 无 | JWT/OAuth2 + RBAC + WAF + 限流 |
| 流式输出 | 否 | SSE 流式首 token |
| 观测 | Trace 面板 + SQLite | OpenTelemetry + Langfuse + Prometheus/Grafana |
| 评测 | 28 条 pytest | 500–1000 条金标集 + Ragas/DeepEval + CI 门禁 |

> 详细的生产化设计见 `技术设计方案_RAG智能客服_v1.md`。

---

## 8. 已知限制

- 当前是**优先级分流 Demo**：P3 低优先可自动回复；P1/P0 兜底并等待人工确认；所有写操作仍是只读/人工。
- 订单/物流/退款是 **Mock 只读数据**，不会执行任何真实写操作。
- 当前知识库是 12 篇种子文档、36 个块；切块规则是简单的 `\n\n`，没有 PDF/Word 上传、表格解析、父子块展开和生效期治理。
- ChromaDB 当前是单机持久化模式，适合 Demo/中小知识库；百万级块或高并发生产建议换 Milvus。
- 向量化和重排默认调用硅基流动 API，需要网络和 API 额度；API 不可用时会回退到 BM25 词法检索。
- 本地 Sentence Transformers 已安装并可加载 reranker，但 `transformers 5.20.2` 下 BGE 小模型 embedding 存在兼容问题；默认走 SiliconFlow。
- 优先级规则是 Demo 级关键词/正则规则，生产环境应改为可配置规则引擎，并用真实工单做分类器评测。
- 优先级分类默认 3 秒超时；超时或模型不可用时自动使用规则分级。
- P3 的 `auto_reply`、P1/P2 的 `need_human` 在模型可用时由 GLM 决定；强制兜底关键词、商品问题、大额订单、资金写操作、投诉不调用模型做降级判断。
- 自动接待语是固定模板，不经过模型；正式上线前请按品牌语气微调。
- 合规/法务措辞是 Demo 级参考，正式上线前需由法务按平台规则、类目和地区复核。
- 未做登录、权限、限流、审计留存策略；不要直接暴露到公网。
- GLM-4.7-Flash 高峰期可能返回 1305 限流；代码已做总超时（默认 8s）、重试、冷却和模板降级，界面会用黄色提示条标记“本次已使用本地模板”。
- `.env` 里的智谱 Key 和硅基流动 Key 都不会进入版本库；如果 Key 曾公开分享，建议在对应控制台轮换。

---

## 9. 建议的下一步

1. 增加 Word/PDF/网页上传与解析，统一按 `\n\n` 切块，并补版本、生效期、owner、蓝绿索引。
2. 建 200–500 条金标集，评测 Recall@15、rerank Top5 命中率、忠实度和端到端延迟。
3. 知识库超过 10 万块或 QPS 上来后，把 ChromaDB 迁移到 Milvus 2.5（dense+sparse 原生混合）。
4. 修复本地 Sentence Transformers 兼容性（固定 `transformers` 版本或换兼容的 BGE 模型），减少对 SiliconFlow API 的依赖。
5. 用真实订单/物流/售后只读 API 替换 `app/data/orders.py`。
6. 增加 Redis 会话与 FAQ 缓存，补流式 SSE。
7. 接客服工作台真实工单系统，打通“采用/编辑/转人工”回写。
8. 增加登录、RBAC、限流、PII 脱敏与审计留存。
9. 生产环境评估生成式 AI 服务备案、算法备案与内容标识要求。
