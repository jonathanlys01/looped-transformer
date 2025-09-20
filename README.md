# Looped Transformer

Looped Transformer experiments codebase

# Setup (using uv)

```bash
uv sync
# --no-cache-dir

uv pip install ninja # (opt)
uv pip install --no-build-isolation flash-attn

uv pip install lighteval[math] # for eval
```


or this one-liner:

```bash
uv sync && uv pip install ninja && uv pip install --no-build-isolation flash-attn && uv pip install lighteval[math]
```
