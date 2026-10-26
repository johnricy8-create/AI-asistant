const state = {
  sessionId: newSessionId(),
  orderId: null,
  orderList: [],
  scenarioList: [],
  sending: false,
  lastResponse: null,
  lastUserMessage: "",
  adopted: false,
  modelName: "",
  autoReply: true,
};

const CHANNEL_META = {
  faq: { label: "FAQ 秒回", className: "faq" },
  rag: { label: "RAG 长尾", className: "rag" },
  tool: { label: "订单工具", className: "tool" },
  human: { label: "转人工", className: "human" },
};

const HOLDING_REPLY = "您好，已收到您的问题，我先帮您查询一下，请稍候。";

const REASON_LABELS = {
  price_protection: "价格保护 / 退差价",
  refund_request: "退款操作",
  complaint: "投诉 / 舆情",
  address_change: "修改收货地址",
  order_cancel: "取消订单",
  user_request: "用户要求人工",
  low_confidence: "检索置信度不足",
  order_not_found: "未查询到订单",
  out_of_scope: "超出商城知识域",
  product_issue: "商品质量问题",
  large_amount: "大额订单售后",
  money_issue: "资金/退款问题",
  priority_high: "高优先级问题",
  invoice: "发票问题",
};

const SKILL_LABELS = {
  after_sales: "售后专席",
  complaint: "投诉专席",
  order_service: "订单服务专席",
  logistics: "物流专席",
  finance: "财务专席",
};

const el = {
  modelDot: document.getElementById("model-dot"),
  modelLabel: document.getElementById("model-label"),
  modelPill: document.getElementById("model-pill"),
  resetBtn: document.getElementById("reset-btn"),
  autoReplyToggle: document.getElementById("auto-reply-toggle"),
  orderSelect: document.getElementById("order-select"),
  customerSubtitle: document.getElementById("customer-subtitle"),
  messages: document.getElementById("messages"),
  scenarioChips: document.getElementById("scenario-chips"),
  input: document.getElementById("message-input"),
  sendBtn: document.getElementById("send-btn"),
  channelBadge: document.getElementById("channel-badge"),
  suggestionEmpty: document.getElementById("suggestion-empty"),
  suggestionBody: document.getElementById("suggestion-body"),
  handoffBanner: document.getElementById("handoff-banner"),
  degradedBanner: document.getElementById("degraded-banner"),
  priorityStrip: document.getElementById("priority-strip"),
  priorityBadge: document.getElementById("priority-badge"),
  priorityHandling: document.getElementById("priority-handling"),
  priorityReason: document.getElementById("priority-reason"),
  confidenceValue: document.getElementById("confidence-value"),
  confidenceBar: document.getElementById("confidence-bar"),
  latencyValue: document.getElementById("latency-value"),
  draft: document.getElementById("draft-text"),
  adoptBtn: document.getElementById("adopt-btn"),
  regenerateBtn: document.getElementById("regenerate-btn"),
  handoffBtn: document.getElementById("handoff-btn"),
  citations: document.getElementById("citations"),
  actions: document.getElementById("actions"),
  riskFlags: document.getElementById("risk-flags"),
  traceLatency: document.getElementById("trace-latency"),
  traceBody: document.getElementById("trace-body"),
  toast: document.getElementById("toast"),
};

let toastTimer = null;

function newSessionId() {
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    const map = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return map[char];
  });
}

function formatTime() {
  return new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatMoney(value) {
  const number = Number(value);
  if (Number.isNaN(number)) return "--";
  return `¥${number.toFixed(2)}`;
}

function formatPercent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

async function api(path, options = {}) {
  const response = await fetch(`/api/v1${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      message = data.detail || JSON.stringify(data);
    } catch (error) {
      const text = await response.text();
      if (text) message = text;
    }
    throw new Error(message);
  }
  return response.json();
}

function showToast(message, duration = 2200) {
  el.toast.textContent = message;
  el.toast.classList.add("show");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.toast.classList.remove("show");
  }, duration);
}

function scrollMessages() {
  requestAnimationFrame(() => {
    el.messages.scrollTop = el.messages.scrollHeight;
  });
}

function renderWelcome() {
  const card = document.createElement("div");
  card.className = "welcome-card";
  card.innerHTML = `
    <h3>顾客小王已接入</h3>
    <p>选择下方场景，或直接输入顾客问题。AI 只生成建议话术，不会自动发送给顾客。</p>
  `;
  el.messages.appendChild(card);
}

function appendMessage(role, content, meta = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}${meta.holding ? " holding" : ""}`;

  if (role !== "system") {
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = role === "customer" ? "王" : "AI";
    wrapper.appendChild(avatar);
  }

  const contentWrapper = document.createElement("div");
  contentWrapper.className = "msg-content";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = content;
  contentWrapper.appendChild(bubble);

  if (meta.holding) {
    const tag = document.createElement("div");
    tag.className = "holding-tag";
    tag.textContent = "自动接待 · 正在为您查询";
    contentWrapper.appendChild(tag);
  }

  if (role !== "system") {
    const time = document.createElement("div");
    time.className = "msg-time";
    time.textContent = meta.time || formatTime();
    contentWrapper.appendChild(time);
  }

  wrapper.appendChild(contentWrapper);
  el.messages.appendChild(wrapper);
  scrollMessages();
}

function appendHandoffCard(result, handoff) {
  const wrapper = document.createElement("div");
  wrapper.className = "message system";
  wrapper.style.display = "block";
  wrapper.style.width = "100%";

  const card = document.createElement("div");
  card.className = "handoff-card";
  const reason = REASON_LABELS[handoff?.reason_code] || handoff?.reason_code || "人工处理";
  const priorityInfo = handoff?.priority_info;
  const skillLabel = SKILL_LABELS[result.skill_group] || result.skill_group || "售后专席";
  card.innerHTML = `
    <strong>已转接 ${escapeHtml(skillLabel)}</strong>
    <p>工单号：${escapeHtml(result.handoff_id)} · 优先级：${escapeHtml(result.priority)}</p>
    <p>转接原因：${escapeHtml(reason)}${
      priorityInfo
        ? ` · 问题等级：${escapeHtml(priorityInfo.label)} · ${escapeHtml(
            priorityInfo.handling_label
          )}`
        : ""
    }</p>
    <p>${escapeHtml(handoff?.summary || "订单与问题摘要已同步给人工客服。")}</p>
  `;
  wrapper.appendChild(card);
  el.messages.appendChild(wrapper);
  scrollMessages();
}

function showTyping() {
  removeTyping();
  const wrapper = document.createElement("div");
  wrapper.className = "message agent";
  wrapper.id = "typing-message";
  wrapper.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-content">
      <div class="msg-bubble typing-bubble">
        <span class="typing"><span></span><span></span><span></span></span>
      </div>
    </div>
  `;
  el.messages.appendChild(wrapper);
  scrollMessages();
}

function removeTyping() {
  const typing = document.getElementById("typing-message");
  if (typing) typing.remove();
}

function setSending(value) {
  state.sending = value;
  el.sendBtn.disabled = value;
  el.regenerateBtn.disabled = value;
}

function updateCustomerSubtitle() {
  const order = state.orderList.find((item) => item.order_id === state.orderId);
  el.customerSubtitle.textContent = order
    ? `订单 ${order.order_id} · ${order.status}`
    : "订单咨询 · 会话已接入";
}

async function init() {
  bindEvents();
  renderWelcome();
  try {
    const [health, scenarios, orders] = await Promise.all([
      api("/health"),
      api("/demo/scenarios"),
      api("/demo/orders"),
    ]);
    renderHealth(health);
    state.scenarioList = scenarios;
    state.orderList = orders;
    renderScenarios(scenarios);
    renderOrders(orders);
  } catch (error) {
    el.modelLabel.textContent = "后端连接失败";
    showToast(`初始化失败：${error.message}`, 4000);
  }
}

function renderHealth(health) {
  const online = health.model_mode === "zhipu";
  state.modelName = health.model || "glm-4.7-flash";
  el.modelDot.classList.toggle("online", online);
  if (online) {
    el.modelLabel.textContent = `${health.model} · 在线`;
  } else {
    el.modelLabel.textContent = "Mock 演示模式";
  }
  el.modelPill.title = online
    ? `模型：${health.model}｜模式：${health.model_mode}`
    : `未配置 ZHIPU_API_KEY，当前使用本地模板；配置后自动调用 ${health.model}。`;
}

function renderScenarios(scenarios) {
  el.scenarioChips.innerHTML = scenarios
    .map(
      (scenario) => `
        <button
          class="chip"
          type="button"
          data-scenario-id="${escapeHtml(scenario.id)}"
          title="${escapeHtml(scenario.description)}"
        >
          ${escapeHtml(scenario.title)}
          <small>${escapeHtml(scenario.hint)}</small>
        </button>
      `
    )
    .join("");
}

function renderOrders(orders) {
  const valid = orders.filter(Boolean);
  if (!valid.length) {
    el.orderSelect.innerHTML = `<option value="">无订单</option>`;
    return;
  }
  el.orderSelect.innerHTML = valid
    .map(
      (order) => `
        <option
          value="${escapeHtml(order.order_id)}"
          title="${escapeHtml(order.order_id)} · ${escapeHtml(
            order.product_name
          )} · ${escapeHtml(order.status)}"
        >
          ${escapeHtml(order.order_id)} · ${escapeHtml(order.product_name)}
        </option>
      `
    )
    .join("");
  state.orderId = valid[0].order_id;
  el.orderSelect.value = state.orderId;
  updateCustomerSubtitle();
}

async function resetSessionState(showMessage = true) {
  try {
    await api("/demo/reset", { method: "POST", body: "{}" });
  } catch (error) {
    // Keep the UI usable even if the reset call fails.
  }
  state.sessionId = newSessionId();
  state.lastResponse = null;
  state.lastUserMessage = "";
  state.adopted = false;
  el.messages.innerHTML = "";
  renderWelcome();
  el.suggestionBody.classList.add("hidden");
  el.suggestionEmpty.classList.remove("hidden");
  el.channelBadge.textContent = "等待提问";
  el.channelBadge.className = "channel-badge";
  el.priorityStrip.classList.add("hidden");
  el.priorityBadge.textContent = "--";
  el.priorityBadge.className = "priority-badge";
  el.priorityHandling.textContent = "--";
  el.priorityHandling.className = "priority-handling";
  el.priorityReason.textContent = "";
  el.degradedBanner.classList.add("hidden");
  el.degradedBanner.textContent = "";
  el.traceLatency.textContent = "--";
  el.traceBody.innerHTML = `<div class="trace-empty">发送问题后展示完整链路</div>`;
  if (showMessage) showToast("已重置演示");
}

async function resetDemo() {
  await resetSessionState(true);
}

async function runScenario(scenario) {
  if (state.sending) return;
  await resetSessionState(false);
  if (scenario.order_id) {
    state.orderId = scenario.order_id;
    el.orderSelect.value = scenario.order_id;
    updateCustomerSubtitle();
  }
  if (scenario.seed_messages && scenario.seed_messages.length) {
    try {
      await api("/demo/seed", {
        method: "POST",
        body: JSON.stringify({
          session_id: state.sessionId,
          messages: scenario.seed_messages,
        }),
      });
      appendMessage("system", "多轮场景已加载 · 承接上文");
      scenario.seed_messages.forEach((message) => {
        appendMessage(
          message.role === "customer" ? "customer" : "agent",
          message.content
        );
      });
    } catch (error) {
      showToast(`加载多轮场景失败：${error.message}`, 3500);
      return;
    }
  }
  el.input.value = scenario.message;
  await sendMessage(scenario.message);
}

async function sendMessage(text, options = {}) {
  const message = String(text || "").trim();
  if (!message || state.sending) return;

  if (!options.regenerate) {
    el.messages.querySelector(".welcome-card")?.remove();
    appendMessage("customer", message);
    appendMessage("agent", HOLDING_REPLY, { holding: true });
    state.lastUserMessage = message;
  }
  el.input.value = "";
  state.adopted = false;
  el.adoptBtn.disabled = false;
  el.adoptBtn.textContent = "采用并发送";
  el.handoffBtn.disabled = false;
  el.handoffBtn.textContent = "转人工";
  setSending(true);
  showTyping();

  try {
    const response = await api("/chat", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        message,
        user_id: "c99001",
        mode: state.autoReply ? "self_service" : "copilot",
        channel: "workbench",
        context: {
          order_id: state.orderId,
          locale: "zh-CN",
        },
        regenerate: Boolean(options.regenerate),
      }),
    });
    removeTyping();
    state.lastResponse = response;
    if (!options.regenerate) state.lastUserMessage = message;
    renderSuggestion(response);
    renderTrace(response);

    if (response.auto_replied && !options.regenerate) {
      appendMessage("agent", response.reply);
      appendMessage(
        "system",
        `低优先级问题已自动回复 · ${response.priority?.label || "P3"}`
      );
      state.adopted = true;
      el.adoptBtn.disabled = true;
      el.adoptBtn.textContent = "已自动回复";
      void api("/feedback", {
        method: "POST",
        body: JSON.stringify({
          message_id: response.message_id,
          type: "adopted",
          comment: "auto_reply",
        }),
      }).catch(() => {});
    }
  } catch (error) {
    removeTyping();
    appendMessage("system", `请求失败：${error.message}`);
    showToast("请求失败，请查看后端日志", 3500);
  } finally {
    setSending(false);
  }
}

function renderSuggestion(data) {
  el.suggestionEmpty.classList.add("hidden");
  el.suggestionBody.classList.remove("hidden");

  const channel = CHANNEL_META[data.channel] || {
    label: data.channel,
    className: "",
  };
  el.channelBadge.textContent = channel.label;
  el.channelBadge.className = `channel-badge ${channel.className}`;

  if (data.model_mode === "zhipu" && state.modelName) {
    el.modelLabel.textContent = `${state.modelName} · 在线`;
    el.modelDot.classList.add("online");
  } else if (data.model_mode === "mock_fallback" && state.modelName) {
    el.modelLabel.textContent = `${state.modelName} · 限流降级`;
    el.modelDot.classList.remove("online");
  }

  const isPriorityFallback =
    data.priority &&
    ["fallback_human", "human"].includes(data.priority.handling);
  if (isPriorityFallback && data.model_mode === "standard" && state.modelName) {
    el.modelLabel.textContent = `${state.modelName} · 规则兜底`;
    el.modelDot.classList.remove("online");
  }

  if (isPriorityFallback) {
    el.confidenceValue.textContent = "兜底";
    el.confidenceBar.style.width = "100%";
    el.confidenceBar.style.background =
      "linear-gradient(90deg, #f59e0b, #ef4444)";
  } else {
    const confidence = Number(data.confidence) || 0;
    el.confidenceValue.textContent = formatPercent(confidence);
    el.confidenceBar.style.width = `${Math.max(
      3,
      Math.round(confidence * 100)
    )}%`;
    if (confidence >= 0.8) {
      el.confidenceBar.style.background =
        "linear-gradient(90deg, #22c55e, #4f46e5)";
    } else if (confidence >= 0.5) {
      el.confidenceBar.style.background =
        "linear-gradient(90deg, #f59e0b, #f97316)";
    } else {
      el.confidenceBar.style.background =
        "linear-gradient(90deg, #ef4444, #f97316)";
    }
  }
  el.latencyValue.textContent = `${data.latency_ms} ms`;
  renderPriority(data.priority);

  el.draft.value = data.reply || "";
  el.adoptBtn.disabled = Boolean(data.auto_replied);
  el.adoptBtn.textContent = data.auto_replied ? "已自动回复" : "采用并发送";
  el.regenerateBtn.disabled = false;
  el.handoffBtn.disabled = false;
  el.handoffBtn.textContent = "转人工";

  if (data.need_human && data.handoff) {
    const reason = REASON_LABELS[data.handoff.reason_code] || data.handoff.reason_code;
    el.handoffBanner.classList.remove("hidden");
    el.handoffBanner.innerHTML = `<strong>建议转人工</strong> · 原因：${escapeHtml(
      reason
    )} · 优先级：${escapeHtml(data.handoff.priority)} · 技能组：${escapeHtml(
      SKILL_LABELS[data.handoff.skill_group] || data.handoff.skill_group
    )}`;
  } else {
    el.handoffBanner.classList.add("hidden");
    el.handoffBanner.innerHTML = "";
  }

  if (data.model_mode === "mock_fallback") {
    el.degradedBanner.classList.remove("hidden");
    el.degradedBanner.textContent =
      "模型限流/超时，本次已使用本地模板生成；建议客服核对后发送。";
  } else {
    el.degradedBanner.classList.add("hidden");
    el.degradedBanner.textContent = "";
  }

  renderCitations(data.citations || []);
  renderActions(data.suggested_actions || []);
  renderRiskFlags(data.risk_flags || []);

  document.querySelectorAll(".feedback-btn").forEach((button) => {
    button.classList.remove("selected");
    button.disabled = false;
  });
}

function renderPriority(priority) {
  if (!priority) {
    el.priorityStrip.classList.add("hidden");
    return;
  }
  el.priorityStrip.classList.remove("hidden");
  const levelClass = String(priority.level || "").toLowerCase();
  el.priorityBadge.textContent = priority.label || priority.level || "--";
  el.priorityBadge.className = `priority-badge ${levelClass}`;
  el.priorityHandling.textContent = priority.handling_label || "--";
  el.priorityHandling.className = `priority-handling ${priority.handling || ""}`;
  el.priorityReason.textContent = [priority.category_label, priority.reason]
    .filter(Boolean)
    .join(" · ");
}

function renderCitations(citations) {
  if (!citations.length) {
    el.citations.innerHTML = `<div class="citation-empty">本次未使用知识库引用（标准话术或人工兜底）。</div>`;
    return;
  }
  el.citations.innerHTML = citations
    .map(
      (citation) => `
        <div class="citation">
          <div class="citation-head">
            <strong>《${escapeHtml(citation.title)}》</strong>
            <span class="score-pill">${escapeHtml(
              Number(citation.score || 0).toFixed(3)
            )}</span>
          </div>
          <div class="citation-section">${escapeHtml(citation.section)}</div>
          <p>${escapeHtml(citation.snippet)}</p>
        </div>
      `
    )
    .join("");
}

function renderActions(actions) {
  if (!actions.length) {
    el.actions.innerHTML = `<span class="risk-empty">无</span>`;
    return;
  }
  el.actions.innerHTML = actions
    .map(
      (action) => `
        <button
          class="action-chip"
          type="button"
          data-action="${escapeHtml(action.type)}"
        >
          ${escapeHtml(action.label)}
        </button>
      `
    )
    .join("");
}

function renderRiskFlags(flags) {
  if (!flags.length) {
    el.riskFlags.innerHTML = `<span class="risk-empty">未发现明显风险</span>`;
    return;
  }
  el.riskFlags.innerHTML = flags
    .map(
      (flag) => `
        <span class="risk-chip ${escapeHtml(flag.level || "warn")}">
          ${escapeHtml(flag.label)}
        </span>
      `
    )
    .join("");
}

function snapshotItem(label, value) {
  if (value === null || value === undefined || value === "") return "";
  return `
    <div class="snapshot-item">
      <label>${escapeHtml(label)}</label>
      <span>${escapeHtml(value)}</span>
    </div>
  `;
}

function renderTrace(data) {
  el.traceLatency.textContent = `${data.latency_ms} ms`;
  const parts = [];

  if (data.order_snapshot) {
    const order = data.order_snapshot;
    parts.push(`
      <div class="snapshot-card">
        <div class="snapshot-title">
          <strong>订单快照</strong>
          <span class="snapshot-status">${escapeHtml(order.status)}</span>
        </div>
        <div class="snapshot-grid">
          ${snapshotItem("订单号", order.order_id)}
          ${snapshotItem("商品", order.product_name)}
          ${snapshotItem("金额", formatMoney(order.amount))}
          ${snapshotItem("下单/支付", order.paid_at)}
          ${snapshotItem("承运商", order.carrier)}
          ${snapshotItem("运单号", order.tracking_no)}
          ${snapshotItem("最新进度", order.latest_event)}
          ${snapshotItem("预计送达", order.estimated_delivery)}
          ${snapshotItem("退款状态", order.refund_status)}
        </div>
      </div>
    `);
  }

  const steps = data.trace || [];
  parts.push(`<div class="trace-section-title">决策链路</div>`);
  if (!steps.length) {
    parts.push(`<div class="debug-note">暂无链路数据</div>`);
  } else {
    parts.push(`
      <div class="trace-steps">
        ${steps
          .map(
            (step) => `
              <div class="trace-step ${escapeHtml(step.status || "ok")}">
                <div class="trace-dot"></div>
                <div class="trace-content">
                  <div class="trace-title">
                    <span>${escapeHtml(step.label)}</span>
                    <span>${escapeHtml(step.latency_ms)}ms</span>
                  </div>
                  <div class="trace-detail">${escapeHtml(step.detail)}</div>
                </div>
              </div>
            `
          )
          .join("")}
      </div>
    `);
  }

  const retrieval = data.debug?.retrieval || [];
  parts.push(`<div class="trace-section-title">检索命中</div>`);
  if (!retrieval.length) {
    parts.push(`<div class="debug-note">本次未走知识库检索。</div>`);
  } else {
    parts.push(`
      <div class="hit-list">
        ${retrieval
          .map(
            (hit) => `
              <div class="hit-card">
                <div class="hit-head">
                  <span>${escapeHtml(hit.title)}</span>
                  <span>${escapeHtml(Number(hit.score || 0).toFixed(3))}</span>
                </div>
                <div class="hit-section">${escapeHtml(hit.section)}</div>
                <div class="hit-snippet">${escapeHtml(hit.snippet)}</div>
                <div class="hit-bar"><span style="width:${Math.max(
                  3,
                  Math.round((Number(hit.score) || 0) * 100)
                )}%"></span></div>
              </div>
            `
          )
          .join("")}
      </div>
    `);
  }

  if (data.debug?.faq) {
    const faq = data.debug.faq;
    parts.push(`
      <div class="debug-note">
        FAQ：${escapeHtml(faq.faq_id)} · top1 ${Number(faq.score).toFixed(
          3
        )} · margin ${Number(faq.margin).toFixed(3)} · 阈值 ${Number(
          faq.threshold
        ).toFixed(2)} · ${faq.passed ? "命中" : "未命中"}
      </div>
    `);
  }

  if (data.debug?.tool) {
    const tool = data.debug.tool;
    parts.push(`
      <div class="debug-note">
        Tool：${escapeHtml(tool.intent)} · 订单 ${escapeHtml(
          tool.order_id || "无"
        )} · ${tool.found ? "查询成功" : "未查询到"}
      </div>
    `);
  }

  if (data.handoff) {
    const reason = REASON_LABELS[data.handoff.reason_code] || data.handoff.reason_code;
    parts.push(`
      <div class="trace-section-title">转人工上下文包</div>
      <div class="handoff-card">
        <strong>${escapeHtml(reason)} · ${escapeHtml(data.handoff.priority)}</strong>
        <p>${escapeHtml(data.handoff.summary)}</p>
        <p>技能组：${escapeHtml(
          SKILL_LABELS[data.handoff.skill_group] || data.handoff.skill_group
        )}</p>
      </div>
    `);
  }

  if (data.priority) {
    parts.push(`
      <div class="debug-note">
        优先级：${escapeHtml(data.priority.label)} · ${escapeHtml(
          data.priority.handling_label
        )} · 来源 ${escapeHtml(data.priority.source)} · ${escapeHtml(
          data.priority.reason
        )}
      </div>
    `);
  }

  parts.push(`
    <div class="debug-note">
      模型：${escapeHtml(data.model)} · ${escapeHtml(data.model_mode)}
      ${
        data.debug?.llm_error
          ? `<br />降级原因：${escapeHtml(data.debug.llm_error)}`
          : ""
      }
    </div>
  `);

  el.traceBody.innerHTML = parts.join("");
}

async function adoptSuggestion() {
  if (!state.lastResponse || state.adopted) return;
  const draft = el.draft.value.trim();
  if (!draft) {
    showToast("建议话术为空");
    return;
  }
  const edited = draft !== state.lastResponse.reply;
  appendMessage("agent", draft);
  state.adopted = true;
  el.adoptBtn.disabled = true;
  el.adoptBtn.textContent = "已采用";
  try {
    await api("/feedback", {
      method: "POST",
      body: JSON.stringify({
        message_id: state.lastResponse.message_id,
        type: edited ? "edited" : "adopted",
        edited_reply: edited ? draft : null,
      }),
    });
    showToast(edited ? "已采用编辑后的建议（演示）" : "已采用并发送（演示）");
  } catch (error) {
    showToast(`反馈记录失败：${error.message}`);
  }
}

async function regenerateSuggestion() {
  if (!state.lastResponse || state.sending) return;
  const message = state.lastUserMessage || state.lastResponse.debug?.normalized_query;
  if (!message) return;
  el.regenerateBtn.disabled = true;
  el.regenerateBtn.textContent = "生成中…";
  try {
    await sendMessage(message, { regenerate: true });
  } finally {
    el.regenerateBtn.disabled = false;
    el.regenerateBtn.textContent = "换一个";
  }
}

async function handoff() {
  if (!state.lastResponse) return;
  const data = state.lastResponse;
  const handoffContext = data.handoff || {
    reason_code: "user_request",
    priority: "normal",
    skill_group: "after_sales",
    summary: `顾客问题：${state.lastUserMessage}`,
  };
  el.handoffBtn.disabled = true;
  el.handoffBtn.textContent = "转接中…";
  try {
    const result = await api("/handoff", {
      method: "POST",
      body: JSON.stringify({
        conversation_id: state.sessionId,
        reason_code: handoffContext.reason_code || "user_request",
        priority: handoffContext.priority || "normal",
        skill_group: handoffContext.skill_group || "after_sales",
        context: handoffContext,
      }),
    });
    appendHandoffCard(result, handoffContext);
    el.handoffBtn.textContent = "已转接";
    showToast("已生成转人工工单（演示）");
  } catch (error) {
    el.handoffBtn.disabled = false;
    el.handoffBtn.textContent = "转人工";
    showToast(`转人工失败：${error.message}`, 3500);
  }
}

async function sendFeedback(type, button) {
  if (!state.lastResponse) return;
  try {
    await api("/feedback", {
      method: "POST",
      body: JSON.stringify({
        message_id: state.lastResponse.message_id,
        type,
        reason_code: type,
      }),
    });
    document.querySelectorAll(".feedback-btn").forEach((item) => {
      item.classList.remove("selected");
    });
    button.classList.add("selected");
    showToast("反馈已记录，感谢");
  } catch (error) {
    showToast(`反馈失败：${error.message}`);
  }
}

function bindEvents() {
  el.sendBtn.addEventListener("click", () => sendMessage(el.input.value));
  el.input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      sendMessage(el.input.value);
    }
  });
  el.adoptBtn.addEventListener("click", adoptSuggestion);
  el.regenerateBtn.addEventListener("click", regenerateSuggestion);
  el.handoffBtn.addEventListener("click", handoff);
  el.resetBtn.addEventListener("click", resetDemo);

  el.orderSelect.addEventListener("change", () => {
    state.orderId = el.orderSelect.value;
    updateCustomerSubtitle();
  });

  el.autoReplyToggle.addEventListener("change", () => {
    state.autoReply = el.autoReplyToggle.checked;
    showToast(
      state.autoReply
        ? "已开启低优先级自动回复"
        : "已切换为纯 Copilot 模式"
    );
  });

  el.scenarioChips.addEventListener("click", (event) => {
    const button = event.target.closest("[data-scenario-id]");
    if (!button) return;
    const scenario = state.scenarioList.find(
      (item) => item.id === button.dataset.scenarioId
    );
    if (scenario) runScenario(scenario);
  });

  el.actions.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    if (button.dataset.action === "handoff") {
      handoff();
    } else {
      showToast("演示版暂未接入该业务页面");
    }
  });

  document.querySelectorAll(".feedback-btn").forEach((button) => {
    button.addEventListener("click", () => {
      sendFeedback(button.dataset.feedback, button);
    });
  });
}

init();
