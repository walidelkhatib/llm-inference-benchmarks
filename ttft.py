import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
prompt = "Write a 200-word explanation of how a car engine works. /no_think"

for i in range(5):
    start = time.perf_counter()
    first = None
    chunks = 0
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            chunks += 1
    end = time.perf_counter()
    # Streamed chunks are usually one token each, so tokens/s is approximate.
    print(f"run {i+1}: TTFT {(first-start)*1000:.0f} ms, "
          f"~{chunks/(end-first):.1f} tokens/s, total {end-start:.2f} s")
