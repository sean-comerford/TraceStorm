# TraceStorm

Install:

```bash
pip install -e .
```

Usage:

Start an OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Run the load test:

```bash
python -m my_project.main \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --trace "0,1.5,3.0" \
    --subprocesses 2
```