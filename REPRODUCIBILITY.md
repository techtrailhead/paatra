# PAATRA Reproducibility Guide

This document describes the evidence currently available for the revised PAATRA manuscript and the steps required to reproduce the reported training curves.

## Scope

The current repository supports:

1. the teacher-only vocabulary screening protocol;
2. the completed 20K-vocabulary student run;
3. the partial 10K-vocabulary student run through step 14,000;
4. regeneration of the reported loss figures from the released logs.

It does not currently support a completed parameter-matched A/B/C comparison because the completed large-vocabulary checkpoint and the original small-budget checkpoints were not recovered.

## Reported configurations

| Run | Vocabulary | Total parameters | Embedding share | Transformer share | Status |
|---|---:|---:|---:|---:|---|
| A: large vocabulary | 47,246 | 49,672,192 | 48.7% | 51.3% | Architecture recovered; completed checkpoint unavailable |
| B: PAATRA 20K | 20,001 | 62,094,912 | 22.7% | 77.3% | Completed 15,000 steps |
| C: PAATRA 10K | 10,001 | 64,778,496 | 11.9% | 88.1% | Partial: 14,000 of 15,000 steps |

Teacher:

```text
Qwen/Qwen2.5-0.5B
494,032,768 parameters
```

Dataset:

```text
Salesforce/wikitext
configuration: wikitext-103-raw-v1
```

Hardware used for the recovered runs:

```text
Single NVIDIA T4 GPU
Google Colab
```

## Environment

```bash
git clone https://github.com/techtrailhead/paatra.git
cd paatra
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The exact software freeze from the original Colab environment has not yet been recovered. Before archival release, run:

```bash
python --version
pip freeze > environment/colab-freeze.txt
nvidia-smi > environment/nvidia-smi.txt
```

## Loading WikiText-103

Use the full Hugging Face repository identifier:

```python
from datasets import load_dataset

corpus = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    split="train",
)
```

Recent Hugging Face Hub versions may reject `load_dataset("wikitext", ...)` because the Hub expects a `namespace/name` repository identifier.

## Recovered training configuration

```yaml
teacher: Qwen/Qwen2.5-0.5B
dataset: Salesforce/wikitext
dataset_config: wikitext-103-raw-v1
sequence_length: 512
batch_size: 4
optimizer: AdamW
peak_learning_rate: 3.0e-4
weight_decay: 1.0e-2
warmup_steps: 500
lr_schedule: cosine
gradient_clip_norm: 1.0
distillation_temperature: 2.0
teacher_loss_weight: 0.7
max_steps: 15000
log_every_steps: 1000
hardware: single NVIDIA T4
```

The exact random seed, precision mode, and gradient-accumulation setting must be recovered from the original notebook and should not be guessed.

## Config B: completed 20K run

Expected structure:

```text
Vocabulary: 20,001
Total parameters: 62,094,912
Embedding share: 22.7%
Transformer share: 77.3%
Training steps: 15,000
Runtime: 143.6 minutes
```

Reference metrics are stored in `logs/B_20K_training.csv`.

Final reported checkpoint values:

```text
Total loss: 2.592
KD loss: 2.074
CE loss: 3.801
```

## Config C: partial 10K run

Expected structure:

```text
Vocabulary: 10,001
Total parameters: 64,778,496
Embedding share: 11.9%
Transformer share: 88.1%
Completed steps: 14,000 of 15,000
Elapsed time: approximately 146 minutes
```

Reference metrics are stored in `logs/C_10K_training_partial.csv`.

Last available values:

```text
Total loss: 2.636
KD loss: 2.186
CE loss: 3.687
```

This is a partial run and must not be reported as a completed 15,000-step result.

## Special-token warning

The recovered 10K configuration inherited:

```text
bos_token_id = 50256
eos_token_id = 50256
```

The valid student vocabulary range is `0` through `10000`.

Before evaluation, set unused special-token IDs to `None` or map them to valid retained IDs:

```python
model.config.bos_token_id = None
model.config.eos_token_id = None

if hasattr(model, "generation_config"):
    model.generation_config.bos_token_id = None
    model.generation_config.eos_token_id = None
```

Validate actual batches:

```python
input_ids = batch["input_ids"]
assert input_ids.min().item() >= 0
assert input_ids.max().item() < model.config.vocab_size
```

## Regenerating figures

```bash
python scripts/plot_training_logs.py
```

Expected output:

```text
figures/loss_B_20K_full.pdf
figures/loss_C_10K_partial.pdf
```

## Supported claims

The current artifact supports the following claims:

- the completed 20K configuration trained for 15,000 steps with decreasing logged losses;
- the partial 10K configuration trained through step 14,000 with decreasing logged losses;
- reduced-vocabulary students with high transformer shares are trainable at approximately 62–65M parameters in the reported setup.

The current artifact does not support:

- a completed parameter-matched A/B/C comparison;
- a causal claim that parameter reallocation improves quality;
- a tokenizer-independent 10.2% gain;
- a final 15,000-step result for Config C.

## Fresh-environment validation still required

Before archival release:

1. load the B checkpoint in a fresh Colab session;
2. reconstruct the exact model without manual edits;
3. verify the exact parameter count;
4. run one forward pass;
5. package the vocabulary mapping with the checkpoint;
6. record exact software versions;
7. recover the seed, precision mode, and gradient-accumulation setting;
8. publish SHA-256 hashes for checkpoints, logs, and vocabulary files.
