from __future__ import annotations

import asyncio

import httpx

from app.config import Settings


async def main() -> None:
    settings = Settings()
    print("model:", settings.zhipu_model)
    print("base_url:", settings.zhipu_base_url)
    print("configured:", bool(settings.zhipu_api_key.strip()))
    print("llm_mode:", settings.llm_mode)
    if not settings.zhipu_api_key.strip():
        print("no api key")
        return

    payload = {
        "model": settings.zhipu_model,
        "messages": [
            {
                "role": "user",
                "content": "请用一句话回复：你好，我是优选商城客服助手。",
            }
        ],
        "temperature": 0.2,
        "max_tokens": 120,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.zhipu_api_key.strip()}",
        "Content-Type": "application/json",
    }
    url = f"{settings.zhipu_base_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        response = await client.post(url, headers=headers, json=payload)
    print("status:", response.status_code)
    print("body:", response.text[:2000])


if __name__ == "__main__":
    asyncio.run(main())
