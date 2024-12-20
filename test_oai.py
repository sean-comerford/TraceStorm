import asyncio
import time
from openai import AsyncOpenAI

K = 10

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)


async def request(request_data: dict):
    result = ""
    token_count = 0
    time_records = [time.time()]
    stream = await client.chat.completions.create(**request_data)

    async for completion in stream:
        tokens = completion.choices[0].delta.content
        time_records.append(time.time())
        token_count += 1  # TODO: Count tokens properly
        result += tokens

    return {
        "result": result,
        "token_count": token_count,
        "time_records": time_records,
    }


async def main() -> None:
    request_data = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
    }
    tasks = [asyncio.create_task(request(request_data)) for _ in range(K)]
    results = await asyncio.gather(*tasks)

    for r in results:
        print(f"Token count: {r['token_count']}")
        print(f"Time records: {r['time_records']}")
        print(f"Result: {r['result'][:100]}")
        print()


asyncio.run(main())
