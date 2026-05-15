# Project Journal — Music CRS Baselines

> **Author:** Sanjay Krishnamurthy  
> **Started:** 2026-05-10  
> **Last updated:** 2026-05-15  
> A running log of findings, hiccups, and resolutions while building the RecSys Challenge 2026 baseline system.

---

## Table of Contents

- [Day 1 — Project Bootstrap (May 10)](#day-1--project-bootstrap-may-10)
- [Day 2 — Local Execution & First Submission (May 14)](#day-2--local-execution--first-submission-may-14)
  - [Hiccup #1: 80 False-Positive Coverage Warnings](#hiccup-1-80-false-positive-coverage-warnings)
  - [Hiccup #2: Submission Upload Fails — "Server returned: 0 error"](#hiccup-2-submission-upload-fails--server-returned-0-error)
- [Day 3 — Package Structure, Kaggle, & OOM Fixes (May 15)](#day-3--package-structure-kaggle--oom-fixes-may-15)
  - [Hiccup #3: `pip install -e .` Doesn't Find Subpackages](#hiccup-3-pip-install--e--doesnt-find-subpackages)
  - [Hiccup #4: Lowerbound Scripts Crash Without `source .env`](#hiccup-4-lowerbound-scripts-crash-without-source-env)
  - [Hiccup #5: Competition Expects `predictions.json`, Not `prediction.json`](#hiccup-5-competition-expects-predictionsjson-not-predictionjson)
  - [Hiccup #6: Kaggle Mount Paths Are Not HuggingFace Hub IDs](#hiccup-6-kaggle-mount-paths-are-not-huggingface-hub-ids)
  - [Hiccup #7: Kaggle Llama Model — PyTorch vs Transformers Format](#hiccup-7-kaggle-llama-model--pytorch-vs-transformers-format)
  - [Hiccup #8: CUDA Out-of-Memory on Kaggle T4 GPU](#hiccup-8-cuda-out-of-memory-on-kaggle-t4-gpu)
  - [Hiccup #9: GitHub Push Auth Expires on Corporate Network](#hiccup-9-github-push-auth-expires-on-corporate-network)
  - [Hiccup #10: `device_map="auto"` Nullified by `.to(device)` — Second GPU Empty](#hiccup-10-device_mapauto-nullified-by-todevice--second-gpu-empty)
- [Lessons Learned](#lessons-learned)

---

## Day 1 — Project Bootstrap (May 10)

**Commit:** `9f7d0c6` — Initial commit

Set up the Music CRS Baselines repository with the two-stage pipeline architecture:

1. **Stage 1 — Retrieval:** BM25 (sparse) or BERT (dense) retrieves top-20 track candidates
2. **Stage 2 — LLM Response:** Llama 3.2 1B generates a conversational response referencing the top-1 track

At this point, the code assumed HuggingFace Hub access for datasets and models — which turned out to be a problem on Walmart's corporate network.

---

## Day 2 — Local Execution & First Submission (May 14)

**Commits:** `1bc1c25` → `dbd0f70`

Added local data support so inference works entirely offline, created `prepare_submission.py` for packaging, and generated the first blind-set submission.

### Hiccup #1: 80 False-Positive Coverage Warnings

**What happened:**  
After running `prepare_submission.py` on the Blind Set A output, I got 80 warnings like:

```
⚠️  Session abc123: missing turns {2, 3, 4, 5, 6, 7, 8} (found: {1})
```

Every single session was flagged. It looked like something was broken.

**Root cause:**  
The blind set is intentionally structured differently from the dev set. The dev set has 8 turns per session (full conversations). The blind set has **80 sessions, each frozen at a different conversation turn** — some at turn 1, some at turn 3, some at turn 7, etc. The distribution:

| Turn | Sessions |
|------|----------|
| 1 | 20 |
| 2 | 15 |
| 3 | 10 |
| 4 | 5 |
| 5 | 8 |
| 6 | 9 |
| 7 | 8 |
| 8 | 5 |

So a session at turn 3 correctly produces 1 prediction at turn 3 — it doesn't need turns 1, 2, 4–8.

**Fix:**  
Updated `check_coverage()` in `prepare_submission.py` to detect single-turn sessions (blind-set mode) and skip the coverage check for them. Only multi-turn sessions with actual gaps within their turn range are flagged.

**Lesson:** Understand your evaluation data structure before writing validation code. Don't assume all datasets follow the same format.

---

### Hiccup #2: Submission Upload Fails — "Server returned: 0 error"

**What happened:**  
After packaging `submission.zip`, uploading to the competition leaderboard returned:

```
Submission upload failed. Server returned: 0 error
```

A "0 error" typically means the request never reached the server at all.

**Root cause:**  
Walmart's corporate VPN was blocking the competition's upload endpoint. The connection was silently dropped.

**Fix:**  
Disconnected from the Walmart VPN and used a personal network (phone hotspot). Upload worked immediately.

**Lesson:** When you get cryptic network errors (status 0, connection reset, timeout), the corporate proxy/VPN is the first suspect. Keep a personal hotspot handy for external submissions.

---

## Day 3 — Package Structure, Kaggle, & OOM Fixes (May 15)

**Commits:** `92bf234` → `3806ee7`

This was the big integration day — fixing the package for proper installation, getting Kaggle GPU inference working, and resolving a cascade of environment issues.

### Hiccup #3: `pip install -e .` Doesn't Find Subpackages

**What happened:**  
After running `pip install -e .`, this import worked fine:

```python
from mcrs import load_crs_baseline  # ✅
```

But these failed:

```python
from mcrs.db_item import MusicCatalogDB      # ❌ ModuleNotFoundError
from mcrs.retrieval_modules import BM25_MODEL  # ❌ ModuleNotFoundError
```

**Root cause:**  
In `pyproject.toml`, the package was declared as:

```toml
[tool.setuptools]
packages = ["mcrs"]          # ← Only installs the top-level mcrs/ directory
```

This tells setuptools to install ONLY `mcrs/` itself, not its subdirectories (`mcrs/db_item/`, `mcrs/retrieval_modules/`, etc.). Those subpackages were simply not included in the installed package.

**Fix:**  
Switched to automatic package discovery:

```toml
[tool.setuptools.packages.find]
include = ["mcrs*"]          # ← Finds mcrs, mcrs.db_item, mcrs.lm_modules, etc.
```

Also added package data for the prompt templates:

```toml
[tool.setuptools.package-data]
mcrs = ["system_prompts/*.txt"]
```

**Lesson:** Always use `packages.find` with a glob pattern for packages with submodules. Explicit `packages = [...]` requires listing every single subpackage manually and breaks silently when you add new ones.

---

### Hiccup #4: Lowerbound Scripts Crash Without `source .env`

**What happened:**  
Running `python lowerbound/random_sample.py` immediately failed with:

```
FileNotFoundError: [Errno 2] No such file or directory:
  'talkpl-ai/TalkPlayData-Challenge-Dataset'
```

The `.env` file had the correct local paths, but I forgot to run `source .env` first.

**Root cause:**  
The scripts use `os.environ.get('MCRS_TEST_DB', 'talkpl-ai/...')` at module level. Without sourcing `.env`, the env vars aren't set, so the default (a HuggingFace Hub ID) is used — which doesn't exist as a local path.

**Fix:**  
Added an auto-dotenv loader in `mcrs/utils.py` that runs on import:

```python
def _load_dotenv() -> None:
    """Read .env from project root into os.environ (stdlib only)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key not in os.environ:      # never overwrite existing vars
                os.environ[key] = value

_load_dotenv()  # runs on first import of mcrs.utils
```

**Critical detail — import ordering matters:**  
The initial fix still broke because of Python's module-level evaluation order:

```python
# ❌ BROKEN — env var read happens BEFORE _load_dotenv()
TRACK_DB = os.environ.get('MCRS_TRACK_DB', '...')   # evaluated immediately
from mcrs.utils import load_local_or_hf_dataset       # _load_dotenv() runs here — too late
```

```python
# ✅ FIXED — import mcrs.utils FIRST
from mcrs.utils import load_local_or_hf_dataset       # _load_dotenv() runs here
TRACK_DB = os.environ.get('MCRS_TRACK_DB', '...')   # now env vars are set
```

**Lesson:** When auto-loading config on import, ensure the import that triggers the load comes BEFORE any code that reads those values. Python evaluates module-level statements top-to-bottom.

---

### Hiccup #5: Competition Expects `predictions.json`, Not `prediction.json`

**What happened:**  
Discovered mid-workflow that the competition evaluator expects the file inside the submission zip to be named `predictions.json` (plural), not `prediction.json` (singular).

**Fix:**  
Renamed everywhere — `prepare_submission.py`, `.gitignore`, all documentation. A global find-and-replace across the project.

**Lesson:** Read the competition spec carefully for exact naming conventions. A single character (`s`) can mean the difference between a valid and rejected submission.

---

### Hiccup #6: Kaggle Mount Paths Are Not HuggingFace Hub IDs

**What happened:**  
Created Kaggle-specific YAML configs that initially used HuggingFace Hub IDs:

```yaml
# ❌ Doesn't work on Kaggle — these are Hub IDs, not file paths
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
lm_type: "meta-llama/Llama-3.2-1B-Instruct"
```

On Kaggle, datasets and models are mounted as local directories, not downloaded from the Hub.

**Root cause:**  
Kaggle mounts attached inputs under `/kaggle/input/`. The exact paths depend on how the user attached the dataset/model:

```
# Data (attached as "sanjaykrish/TalkPlayData_New" dataset)
/kaggle/input/datasets/sanjaykrish/talkplaydata-new/data/raw/TalkPlayData-Challenge-Track-Metadata

# Model (attached as Llama 3.2 → Transformers → 1B Instruct)
/kaggle/input/models/metaresearch/llama-3.2/transformers/1b-instruct/1
```

**Fix:**  
Created separate `kaggle_*` YAML configs with the actual Kaggle mount paths:

```yaml
lm_type: "/kaggle/input/models/metaresearch/llama-3.2/transformers/1b-instruct/1"
item_db_name: "/kaggle/input/datasets/sanjaykrish/talkplaydata-new/data/raw/TalkPlayData-Challenge-Track-Metadata"
```

**Lesson:** When running on Kaggle, always verify mount paths from the Kaggle file browser (left sidebar) before configuring. Don't assume slug-casing or directory structure.

---

### Hiccup #7: Kaggle Llama Model — PyTorch vs Transformers Format

**What happened:**  
Initially attached the Llama 3.2 model on Kaggle using the **PyTorch** framework variant. The mounted directory contained:

```
consolidated.00.pth
params.json
```

Our code uses `AutoModelForCausalLM.from_pretrained()` from HuggingFace Transformers, which expects:

```
config.json
model.safetensors (or model-00001-of-*.safetensors)
tokenizer.json
tokenizer_config.json
```

**Root cause:**  
When attaching a model on Kaggle, you choose the **framework**: PyTorch, Transformers, JAX, etc. The PyTorch variant ships raw `.pth` checkpoint files. The Transformers variant ships HF-compatible files that `from_pretrained()` can load directly.

**Fix:**  
Switched to the **Transformers** framework variant when attaching the model on Kaggle:

1. Kaggle sidebar → **+ Add Input** → **Models**
2. Search: `meta-llama/llama-3.2`
3. Select: **Llama 3.2** → ⚠️ **Transformers** (not PyTorch) → **1B Instruct** → **Version 1**

Mount path became: `/kaggle/input/models/metaresearch/llama-3.2/transformers/1b-instruct/1`

**Lesson:** Always match the model framework to your loading code. `from_pretrained()` = Transformers format. `torch.load()` = PyTorch format. Kaggle makes you choose upfront.

---

### Hiccup #8: CUDA Out-of-Memory on Kaggle T4 GPU

**What happened:**  
Running Llama 3.2 1B inference on Kaggle's T4 GPU blew up:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 858.00 MiB.
GPU 0 has a total capacity of 14.56 GiB,
of which 218.44 MiB is free.
```

Command was: `python run_inference_blindset.py --tid kaggle_llama1b_bm25_blindset_A --batch_size 8`

**Root cause — three contributing factors:**

1. **bfloat16 on T4:** The code defaulted to `torch.bfloat16`, but T4 GPUs don't natively support bfloat16. PyTorch silently emulates it using float32 operations under the hood, roughly **doubling** actual memory usage compared to native float16.

2. **No input length cap:** Long conversation histories (8 turns of dialogue + system prompt + retrieval context) produced token sequences well beyond 2048 tokens. With batch_size=8, the padded tensor consumed enormous VRAM.

3. **Eager model loading:** Without `low_cpu_mem_usage=True`, the model was loaded fully into CPU RAM before being transferred to GPU, causing a peak memory spike during the `.to(device)` call.

**Fix — three changes to `mcrs/lm_modules/llama.py`:**

```python
# 1. Auto-detect T4 and downgrade dtype
if device == "cuda" and dtype == torch.bfloat16:
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("[INFO] GPU does not support bfloat16 — using float16 instead.")
        dtype = torch.float16

# 2. Load model efficiently
lm = AutoModelForCausalLM.from_pretrained(
    self.model_name,
    attn_implementation=self.attn_implementation,
    torch_dtype=self.dtype,        # pass dtype at load time
    low_cpu_mem_usage=True,        # sequential loading
)

# 3. Cap tokenization length
token_inputs = self.tokenizer(
    formatted_chats,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=2048,               # prevent unbounded sequence length
)
```

Also recommended reducing `--batch_size` to 1 or 2 for the T4's 14.5 GiB VRAM.

**Memory budget (approximate):**

| Component | VRAM |
|-----------|------|
| Llama 3.2 1B weights (float16) | ~2 GB |
| KV cache (1 seq × 2048 tokens) | ~0.5 GB |
| Batch of 8 × padded input tensors | ~3–6 GB |
| Activation memory during generation | ~2–4 GB |
| PyTorch CUDA allocator overhead | ~1 GB |
| **Total (batch=8)** | **~9–14 GB** → OOM |
| **Total (batch=1)** | **~4–5 GB** → fits easily |

**Lesson:** T4 GPUs are deceptive — they advertise 16 GB but bfloat16 emulation halves your effective capacity. Always check `torch.cuda.is_bf16_supported()` and fall back to float16. And always set a `max_length` on tokenization for production inference.

---

### Hiccup #9: GitHub Push Auth Expires on Corporate Network

**What happened:**  
Multiple `git push` attempts failed with:

```
remote: Invalid username or token.
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

**Root cause:**  
The GitHub Personal Access Token (PAT) cached in the macOS credential helper had expired. Corporate security policies enforce short-lived tokens.

**Fix options:**

1. **Re-generate PAT** at [github.com/settings/tokens](https://github.com/settings/tokens) and update:
   ```bash
   git remote set-url origin https://<USERNAME>:<NEW_PAT>@github.com/<repo>.git
   ```

2. **Use SSH instead** (if port 22 isn't blocked by corporate firewall):
   ```bash
   git remote set-url origin git@github.com:<USERNAME>/<repo>.git
   ```

3. **Use `gh auth login`** (GitHub CLI handles token refresh):
   ```bash
   gh auth login
   git push origin main
   ```

**Lesson:** If you're on a corporate network, use `gh auth login` for GitHub CLI-managed tokens, or SSH keys. HTTPS PATs expire silently and block pushes at the worst time.

---

### Hiccup #10: `device_map="auto"` Nullified by `.to(device)` — Second GPU Empty

**What happened:**  
After applying the Hiccup #8 OOM fixes (float16, truncation, `low_cpu_mem_usage`), the T4 x2 Kaggle setup still OOM'd:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 430.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 407.81 MiB is free.
```

The key clue: **GPU 0 had 14.16 GiB in use, but GPU 1 was completely empty.** Kaggle provides two T4 GPUs (2 × 14.56 GiB = ~29 GiB total), but the model was only on one.

**Root cause — `.to(device)` undoes `device_map="auto"`:**

```python
# In _load_model():
lm = AutoModelForCausalLM.from_pretrained(
    ...,
    device_map="auto",    # ✅ Distributes model across GPU 0 + GPU 1
)

# In __init__(), immediately after:
self.lm.to(self.device)   # ❌ MOVES EVERYTHING BACK TO GPU 0!
```

When HuggingFace's `device_map="auto"` is used, it assigns each layer to a device based on available memory. But calling `.to("cuda")` afterwards overrides all placements and moves every parameter to `cuda:0`. The second GPU stays empty.

**Additional issues found:**

1. **Hardcoded `self.device` for input tensors** — `input_ids.to(self.device)` sent inputs to `"cuda"` (= `cuda:0`), but with `device_map="auto"` the model's first layer might be on any device.

2. **No CUDA cache cleanup** — Without `torch.cuda.empty_cache()` between generations, VRAM fragmentation accumulated over 80 inference calls.

3. **`max_new_tokens=512` for single inference** — way more tokens than needed for a music recommendation response, wasting KV cache memory.

4. **Missing `accelerate` dependency** — `device_map="auto"` silently requires the `accelerate` library. Without it, the model loads onto a single device.

**Fix — five changes:**

```python
# 1. REMOVED: self.lm.to(self.device).to(self.dtype)
# The model is already placed correctly by device_map="auto"

# 2. Added input_device property to find the model's actual first device
@property
def input_device(self):
    return next(self.lm.parameters()).device

# 3. Use input_device for tensor placement (not self.device)
target_device = self.input_device
input_ids = token_inputs.input_ids.to(target_device)

# 4. Clear VRAM after each generation to reduce fragmentation
torch.cuda.empty_cache()

# 5. Set PYTORCH_CUDA_ALLOC_CONF at module level
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

Also added `accelerate>=0.26.0` to `pyproject.toml` dependencies and reduced `max_new_tokens` from 512 to 128 for single-sample inference.

**Why `device_map="auto"` + `accelerate` works:**

| Layer Group | GPU 0 | GPU 1 |
|------------|-------|-------|
| Embedding + layers 0–11 | ✅ | |
| Layers 12–22 + LM head | | ✅ |
| **VRAM per GPU** | **~7–8 GB** | **~7–8 GB** |
| **Free per GPU** | **~6–7 GB** | **~6–7 GB** |

With ~13 GB free across both GPUs for KV cache + activations + batch tensors, inference with `batch_size=2` fits comfortably.

**Lesson:** `device_map="auto"` and `.to(device)` are mutually exclusive strategies. If you use `device_map="auto"`, never call `.to()` on the model afterwards — it undoes the distribution. And always check that `accelerate` is installed, since `device_map` is a no-op without it.

---

## Lessons Learned

### Environment & Infrastructure

1. **Corporate VPN blocks everything** — HuggingFace downloads, competition uploads, GitHub pushes. Have a personal hotspot ready and use local/cached data where possible.

2. **Kaggle is free GPU gold** — T4 GPUs, 30 hrs/week, pre-mounted models with no download wait. But you must understand mount paths and framework formats.

3. **Auto-load your .env** — Don't rely on users remembering `source .env`. A stdlib-only dotenv loader in your package init is zero-dependency and prevents cryptic `FileNotFoundError` crashes.

### Python Packaging

4. **Use `packages.find` not `packages = [...]`** — Explicit package lists break silently when you add subpackages. `include = ["mcrs*"]` catches everything.

5. **Import order is evaluation order** — If your auto-config runs on import, make sure that import statement comes before any code that reads the config.

### GPU & Model Inference

6. **T4 doesn't support bfloat16 natively** — PyTorch silently emulates it at 2× memory cost. Always check `torch.cuda.is_bf16_supported()` and fall back to float16.

7. **Always cap tokenization length** — Without `max_length`, a single long conversation can produce a 4000+ token sequence that blows up your batch. Set `max_length=2048` as a safety net.

8. **`low_cpu_mem_usage=True` is free performance** — Loads model weights sequentially instead of all at once. No quality impact, lower peak memory.

9. **batch_size=1 or 2 is fine for inference** — On constrained GPUs, reducing batch size is the simplest OOM fix. The quality is identical; only wall-clock time increases.

10. **`device_map="auto"` and `.to(device)` are enemies** — If you use `device_map="auto"` to distribute a model across multiple GPUs, calling `.to("cuda")` afterwards moves everything back to a single GPU. Always use `next(model.parameters()).device` to find where to send inputs.

11. **`accelerate` is a silent dependency** — `device_map="auto"` in Transformers requires the `accelerate` library. Without it, the model loads onto a single device with no warning. Always add it to your dependencies explicitly.

### Competition-Specific

10. **Read the spec twice** — `prediction.json` vs `predictions.json` cost me a rename across the entire project. Small naming errors cause silent submission rejections.

11. **Understand your evaluation data** — The blind set has 1 turn per session (not 8). Building validation logic without understanding this structure generated 80 false warnings that looked like real errors.

14. **Cache invalidation is real** — Changed `corpus_types` but forgot to delete `./cache`. Got stale BM25 indexes returning wrong results. The inference scripts now auto-delete cache on startup, but be aware if you run custom experiments.

---

## Timeline Summary

| Date | Milestone | Key Commits |
|------|-----------|-------------|
| May 10 | Project bootstrap, initial two-stage pipeline | `9f7d0c6` |
| May 14 | Local data support, first blind-set submission, Kaggle configs | `1bc1c25` → `dbd0f70` |
| May 15 | Package fix, auto-dotenv, Kaggle mount paths, T4 OOM fix, multi-GPU fix | `92bf234` → latest |

**Total hiccups resolved:** 10  
**Coffees consumed:** ☕☕☕☕ (estimated, trending up)
