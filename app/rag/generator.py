from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.core.text import first_sentence


@dataclass
class GeneratorResult:
    text: str
    model: str
    mode: str  # standard | mock | zhipu | mock_fallback
    error: str | None = None


SYSTEM_PROMPT = """你是优选商城的客服 Copilot，服务对象是一线客服，不是顾客本人。
你的任务是根据提供的订单信息、工具查询结果和知识库参考资料，生成一段可以直接发给顾客的中文客服话术。

硬性要求：
1. 只能依据<参考资料>和<订单/工具结果>回答，不得编造政策、时效、金额或物流节点。
2. 涉及定金、退款、价保、退货时，必须区分适用条件与例外（例如“个人原因”与“商家原因”、“未支付尾款”与“已支付尾款”），不得用“通常不可退”“一律不退”等笼统结论。
3. 如果提供了订单快照，可以结合订单状态给出下一步建议，但不得编造订单信息。
4. 不得承诺退款、赔偿、退差价、改地址等写操作；涉及具体写操作时建议转人工。
5. 纯规则解释不需要转人工；只有资料不足、涉及具体操作或高风险时才建议转人工。
6. 语气礼貌、简洁、先结论后步骤；控制在 2–4 句话，不超过 220 字。
7. 不要输出 Markdown、JSON、来源列表或“作为AI”等自我指涉。
8. 表达委婉、合规：不承诺“一定退款/一定赔偿/一定退差价”，不使用“肯定能退/不能退”等绝对结论；涉及规则、时效、金额时说明“具体以平台规则和审核结果为准”。
9. 涉及商品问题、售后争议或资金操作时，建议顾客保留商品、包装和相关凭证，并建议由人工进一步核实。
"""

PRIORITY_SYSTEM_PROMPT = """你是电商客服工单分级器。请根据顾客问题、订单快照和规则预判，输出严格 JSON。

分级标准：
- P0 紧急：人身安全、食品安全、诈骗/盗刷、账户被盗、法律/监管、媒体曝光、重大舆情。
- P1 高优先：退款/退差价/退定金/赔偿等写操作、商品质量/破损/错漏发/假货、大额订单售后、投诉。
- P2 中优先：物流异常/延迟、退货换货流程、退款进度、发票、改地址、订单状态、预售规则。
- P3 低优先：一般规则解释、运费/包邮、发货时效、会员权益、问候、流程咨询。

判断要求：
1. 商品质量问题（质量、破损、故障、错漏发、假货、过期、变质、安全问题）必须标记 product_issue=true，优先级至少 P1。
2. 订单金额 >= 500 且涉及售后/退款/商品/投诉时，标记 large_amount=true，优先级至少 P1。
3. 只有“我要退款/退差价/退定金/赔偿/补偿”等写操作才标记 money_action=true；“退款到账了吗/退款进度”属于状态查询，不是写操作。
4. 涉及写操作、商品问题、大额订单、投诉时，need_human 必须为 true。
5. 只有 P3 低风险、资料充分、可以直接回答时，auto_reply 才为 true；P1/P2/P0 不要 auto_reply。
6. P1/P2 的纯规则问题，由你根据问题类型判断 need_human：能明确用规则回答则 false，需要核实订单/操作/争议则 true。
7. 不要为了省事把高风险问题降为 P3。

只输出以下 JSON，不要输出 Markdown 或解释：
{
  "priority": "P0|P1|P2|P3",
  "category": "资金安全|商品质量|退款售后|物流|发票|账号|一般咨询|其他",
  "product_issue": true,
  "money_issue": true,
  "money_action": true,
  "large_amount": true,
  "complaint": true,
  "need_human": true,
  "auto_reply": false,
  "reason": "一句话说明分级依据",
  "confidence": 0.0
}
"""

PRIORITY_LEVELS = {"P0", "P1", "P2", "P3"}


def _parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _normalise_priority_data(data: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(data.get("priority") or "").upper()
    match = re.search(r"P[0-3]", raw)
    if not match:
        return None
    data["priority"] = match.group(0)
    for key in [
        "product_issue",
        "money_issue",
        "money_action",
        "large_amount",
        "complaint",
        "need_human",
        "auto_reply",
    ]:
        value = data.get(key)
        if isinstance(value, str):
            data[key] = value.strip().lower() in {"true", "1", "yes", "是"}
        else:
            data[key] = bool(value)
    try:
        data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
    except (TypeError, ValueError):
        data["confidence"] = 0.75
    return data


class GeneratorService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cooldown_until = 0.0

    async def generate(
        self,
        *,
        channel: str,
        question: str,
        faq: Any | None,
        hits: list[dict[str, Any]],
        tool: dict[str, Any] | None,
        order_snapshot: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> GeneratorResult:
        if channel == "faq" and faq is not None:
            return GeneratorResult(text=faq.answer, model="faq-standard", mode="standard")

        if channel == "human":
            return GeneratorResult(
                text=(
                    "您好，这个问题涉及退款、差价或投诉等需要人工核实的操作。"
                    "我已为您转接售后专席，并把订单信息和问题摘要一起带过去了，请稍候。"
                ),
                model="handoff-template",
                mode="standard",
            )

        if self.settings.zhipu_enabled and time.monotonic() < self._cooldown_until:
            fallback = self._mock(
                channel=channel,
                question=question,
                hits=hits,
                tool=tool,
            )
            return GeneratorResult(
                text=fallback,
                model="mock-template",
                mode="mock_fallback",
                error="智谱接口限流冷却中，已使用本地模板",
            )

        if self.settings.zhipu_enabled:
            try:
                text = await self._call_zhipu(
                    channel=channel,
                    question=question,
                    hits=hits,
                    tool=tool,
                    order_snapshot=order_snapshot,
                    history=history,
                )
                return GeneratorResult(
                    text=text,
                    model=self.settings.zhipu_model,
                    mode="zhipu",
                )
            except Exception as exc:  # noqa: BLE001 - degrade gracefully for the demo
                fallback = self._mock(
                    channel=channel,
                    question=question,
                    hits=hits,
                    tool=tool,
                )
                return GeneratorResult(
                    text=fallback,
                    model="mock-template",
                    mode="mock_fallback",
                    error=f"{type(exc).__name__}: {exc}"[:240],
                )

        return GeneratorResult(
            text=self._mock(channel=channel, question=question, hits=hits, tool=tool),
            model="mock-template",
            mode="mock",
        )

    async def classify_priority(
        self,
        *,
        question: str,
        order_snapshot: dict[str, Any] | None,
        risk_flags: list[dict[str, str]],
        intent: str,
    ) -> dict[str, Any] | None:
        """LLM-based priority classification.

        Rules already provide a floor in PriorityEngine. This call is only made
        for P2/P3 rule candidates, so a failure is safe and falls back to rules.
        """
        if not self.settings.zhipu_enabled:
            return None
        if time.monotonic() < self._cooldown_until:
            return None

        risk_labels = "、".join(flag.get("label", "") for flag in risk_flags) or "无"
        user_prompt = f"""<顾客问题>
{question}
</顾客问题>

<订单快照>
{json.dumps(order_snapshot, ensure_ascii=False) if order_snapshot else "（无）"}
</订单快照>

<规则预判>
意图：{intent}
风险标记：{risk_labels}
</规则预判>

请输出分级 JSON。"""

        try:
            text = await self._chat(
                system_prompt=PRIORITY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=300,
                temperature=0.0,
                total_timeout_seconds=self.settings.priority_total_timeout_seconds,
            )
        except Exception:  # noqa: BLE001 - classifier is best-effort
            return None

        data = _parse_json_object(text)
        if not data:
            return None
        return _normalise_priority_data(data)

    def _mock(
        self,
        *,
        channel: str,
        question: str,
        hits: list[dict[str, Any]],
        tool: dict[str, Any] | None,
    ) -> str:
        if channel == "tool" and tool:
            return self._mock_tool(tool, hits)
        return self._mock_rag(question, hits)

    def _mock_tool(self, tool: dict[str, Any], hits: list[dict[str, Any]]) -> str:
        base = tool.get("answer") or "已为您查询订单信息。"
        if not hits:
            return base

        top = hits[0]
        policy = first_sentence(top["content"], 160)
        policy_text = f"根据《{top['title']}》{top['section']}：{policy}"
        intent = tool.get("intent")
        if intent == "refund_status":
            tail = "如超过预计时间仍未到账，我可以帮您转人工核实，具体以支付机构处理结果为准。"
        elif intent == "logistics":
            tail = "如物流长时间没有更新，我可以帮您提交物流异常或转人工跟进，具体以承运商和平台核实结果为准。"
        else:
            tail = "如果您需要，我可以帮您催发货或转人工进一步处理，具体以平台规则和实际处理结果为准。"
        return f"{base}{policy_text}{tail}"

    def _mock_rag(self, question: str, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return (
                "您好，我暂时没有检索到足够明确的规则，为避免给您错误信息，"
                "建议转人工客服进一步核实。"
            )

        top = hits[0]
        if "预售" in top["title"] and "定金" in question:
            return (
                "您好，预售定金能否退还需结合订单状态、尾款支付情况和平台规则核实："
                "因商家原因导致无法发货、商品缺货等情形，可以申请退还；个人原因未支付尾款一般不支持退还。"
                "具体以平台审核结果为准。您这笔订单目前是「待发货」，需要我帮您转人工核实吗？"
            )

        body = first_sentence(top["content"], 260)
        text = f"您好，根据《{top['title']}》{top['section']}：{body}"
        if len(hits) > 1:
            second = hits[1]
            if second["doc_id"] != top["doc_id"]:
                text += f"另外，{first_sentence(second['content'], 180)}"
        text += "以上规则供您参考，具体以平台规则和审核结果为准；如订单情况特殊，我可以帮您转人工进一步核实。"
        return text

    async def _call_zhipu(
        self,
        *,
        channel: str,
        question: str,
        hits: list[dict[str, Any]],
        tool: dict[str, Any] | None,
        order_snapshot: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> str:
        context_blocks = []
        for index, hit in enumerate(hits, start=1):
            context_blocks.append(
                f"[{index}] 《{hit['title']}》 {hit['section']}\n{hit['content']}\n来源：{hit.get('source_uri', '')}"
            )
        context = "\n\n".join(context_blocks) or "（无检索资料）"

        history_text = "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in history[-self.settings.max_history_turns :]
        ) or "（无历史会话）"

        user_prompt = f"""<顾客问题>
{question}
</顾客问题>

<会话摘要>
{history_text}
</会话摘要>

<订单快照>
{json.dumps(order_snapshot, ensure_ascii=False) if order_snapshot else "（无）"}
</订单快照>

<工具结果>
{json.dumps(tool, ensure_ascii=False) if tool else "（无）"}
</工具结果>

<参考资料>
{context}
</参考资料>

通道：{channel}
请生成一段客服可以直接发送给顾客的话术。"""

        return await self._chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self.settings.zhipu_max_tokens,
            temperature=0.2,
        )

    async def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        total_timeout_seconds: float | None = None,
    ) -> str:
        if not self.settings.zhipu_enabled:
            raise RuntimeError("未配置智谱 API Key")

        payload = {
            "model": self.settings.zhipu_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.settings.zhipu_disable_thinking:
            payload["thinking"] = {"type": "disabled"}

        headers = {
            "Authorization": f"Bearer {self.settings.zhipu_api_key.strip()}",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.zhipu_base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None
        max_attempts = max(1, self.settings.zhipu_max_retries + 1)
        total_timeout = (
            total_timeout_seconds
            if total_timeout_seconds is not None
            else self.settings.zhipu_total_timeout_seconds
        )
        deadline = time.monotonic() + total_timeout

        for attempt in range(max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0.1:
                raise last_error or RuntimeError("智谱调用超过总超时时间")
            try:
                async with httpx.AsyncClient(
                    timeout=min(self.settings.zhipu_timeout_seconds, remaining)
                ) as client:
                    response = await asyncio.wait_for(
                        client.post(url, headers=headers, json=payload),
                        timeout=remaining,
                    )
            except (asyncio.TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
                self._cooldown_until = (
                    time.monotonic() + self.settings.zhipu_cooldown_seconds
                )
                last_error = RuntimeError(
                    f"智谱接口连接失败：{type(exc).__name__}: {exc}"
                )
                if attempt < max_attempts - 1:
                    sleep_for = min(
                        self.settings.zhipu_retry_backoff_seconds * (attempt + 1),
                        max(0.0, deadline - time.monotonic()),
                    )
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                        continue
                raise last_error

            if response.status_code == 429 or 500 <= response.status_code < 600:
                self._cooldown_until = (
                    time.monotonic() + self.settings.zhipu_cooldown_seconds
                )
                last_error = RuntimeError(
                    f"智谱接口暂时不可用：HTTP {response.status_code} {response.text[:200]}"
                )
                if attempt < max_attempts - 1:
                    sleep_for = min(
                        self.settings.zhipu_retry_backoff_seconds * (attempt + 1),
                        max(0.0, deadline - time.monotonic()),
                    )
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                        continue
                raise last_error

            response.raise_for_status()
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            if content:
                self._cooldown_until = 0.0
                return content
            usage = data.get("usage") or {}
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            raise RuntimeError(
                "智谱返回空内容"
                f"（finish_reason={choice.get('finish_reason')},"
                f" reasoning_tokens={reasoning_tokens}）"
            )

        raise last_error or RuntimeError("智谱调用失败")
