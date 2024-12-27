# TraceStorm

Install:

```bash
pip install tracestorm
```

Usage:

Start an OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Run the load test:

```bash
python -m trace_storm.main --model "Qwen/Qwen2.5-1.5B-Instruct"
```