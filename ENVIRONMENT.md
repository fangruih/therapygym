# TherapyGym Environment Setup

This document describes the environment setup and known compatibility requirements for running TherapyGym training.

## Quick Start (Recommended)

Use the setup script to handle all dependencies automatically:

```bash
# Run the setup script (takes ~15-20 minutes due to flash-attn compilation)
./setup_env.sh

# Activate the environment
source .venv/bin/activate

# Run training
NUM_GPUS=4 TRAIN_BATCH_SIZE=8 bash scripts/train_therapy.sh
```

## Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment and install dependencies with uv
uv sync --python 3.10

# Activate the virtual environment
source .venv/bin/activate

# Install flash-attn (requires CUDA, builds from source ~10-15 min)
uv pip install "flash-attn>=2.6.0,<2.7.0" --no-build-isolation

# Upgrade OpenTelemetry for Ray compatibility
uv pip install "opentelemetry-api>=1.27" "opentelemetry-sdk>=1.27" "opentelemetry-exporter-prometheus>=0.48b0"

# Run training
NUM_GPUS=4 TRAIN_BATCH_SIZE=8 bash scripts/train_therapy.sh
```

## Critical Version Compatibility

The following packages have strict version dependencies that must be respected:

| Package | Required Version | Reason |
|---------|------------------|--------|
| `torch` | `>=2.6.0,<2.7.0` | vllm 0.8.x requires torch 2.6.x |
| `torchvision` | `>=0.21.0,<0.22.0` | Must match torch 2.6.x |
| `torchaudio` | `>=2.6.0,<2.7.0` | Must match torch 2.6.x |
| `vllm` | `>=0.8.4,<0.9.0` | verl 0.3.x is incompatible with vllm 0.13+ |
| `flash-attn` | `>=2.6.0,<2.7.0` | Must be built against the installed torch version |
| `opentelemetry-api` | `>=1.27.0` | Ray 2.53+ requires newer OpenTelemetry |
| `opentelemetry-sdk` | `>=1.27.0` | Ray 2.53+ requires newer OpenTelemetry |
| `setuptools` | `<81` | verl uses deprecated `pkg_resources` |

## Common Issues and Solutions

### 1. `AttributeError: module 'torch' has no attribute 'utils'`

**Cause:** Python is not using the virtual environment's torch.

**Solution:** Always activate the virtual environment before running:
```bash
source .venv/bin/activate
```

### 2. `RuntimeError: operator torchvision::nms does not exist`

**Cause:** Version mismatch between torch and torchvision.

**Solution:** Ensure torchvision matches torch version:
```bash
uv pip install --upgrade torchvision torchaudio
```

### 3. `ModuleNotFoundError: No module named 'flash_attn'`

**Cause:** flash-attn is not installed.

**Solution:** Install flash-attn (requires CUDA):
```bash
uv pip install flash-attn --no-build-isolation
```

### 4. `ImportError: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb` (vllm)

**Cause:** vllm was compiled against a different torch version.

**Solution:** Reinstall vllm or downgrade to compatible version:
```bash
uv pip install "vllm==0.8.4"
```

### 5. `ModuleNotFoundError: No module named 'vllm.worker'`

**Cause:** verl is incompatible with the installed vllm version.

**Solution:** verl 0.3.x requires vllm 0.8.x. Downgrade vllm:
```bash
uv pip install "vllm==0.8.4"
```

### 6. `ImportError: undefined symbol` (flash_attn)

**Cause:** flash-attn was compiled against a different torch version.

**Solution:** Rebuild flash-attn from source:
```bash
uv pip uninstall flash-attn
uv pip install "flash-attn>=2.6.0,<2.7.0" --no-build-isolation
```
Note: Building from source can take 10-30 minutes.

### 7. `TypeError: Meter.create_histogram() got an unexpected keyword argument`

**Cause:** OpenTelemetry version is too old for Ray.

**Solution:** Upgrade OpenTelemetry:
```bash
uv pip install --upgrade "opentelemetry-api>=1.27" "opentelemetry-sdk>=1.27"
```

## Verified Working Configuration

The following package versions are known to work together (as of January 2026):

```
torch                    2.6.0+cu124
torchvision              0.21.0
torchaudio               2.6.0
vllm                     0.8.4
flash-attn               2.6.3
verl                     0.3.1.dev0
ray                      2.53.0
opentelemetry-api        1.39.1
opentelemetry-sdk        1.39.1
transformers             4.57.3
```

## Environment Variables

The training script sets these environment variables automatically:

```bash
# vLLM configuration
VLLM_ATTENTION_BACKEND=FLASH_ATTN
VLLM_USE_V1=1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Cache directories (avoid home directory quota issues)
HF_HOME="${BASE_DIR}/.cache/huggingface"
VLLM_CACHE_ROOT="${BASE_DIR}/.cache/vllm"
```

## GPU Requirements

- Minimum: 4x GPUs with 40GB+ VRAM each (e.g., A100-40GB)
- Recommended: 4x A100-80GB for larger batch sizes
- CUDA 12.4+ required for the pinned package versions
