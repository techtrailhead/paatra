# PAATRA

**Parameter Allocation Aware Training for Reasoning Augmentation**

PAATRA studies a simple design question for small language models: when the total parameter budget is limited, how much capacity should be assigned to the vocabulary matrices and how much should be assigned to transformer blocks?

The project constructs frequency-selected subsets of a teacher vocabulary and reallocates saved embedding parameters into the student transformer.

## Current artifact status

This repository currently documents the recovered experimental evidence:

| Run | Vocabulary | Parameters | Transformer share | Status |
|---|---:|---:|---:|---|
| A: large vocabulary | 47,246 | 49,672,192 | 51.3% | Architecture recovered; completed checkpoint unavailable |
| B: PAATRA 20K | 20,001 | 62,094,912 | 77.3% | Completed 15,000 steps |
| C: PAATRA 10K | 10,001 | 64,778,496 | 88.1% | Partial run through step 14,000 |

The available runs are **not parameter matched**. Therefore, this repository does not claim that the recovered B/C comparison establishes a causal allocation advantage.

## Teacher and dataset

- Teacher: `Qwen/Qwen2.5-0.5B`
- Dataset: `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`
- Hardware used for recovered runs: one NVIDIA T4 GPU in Google Colab

## Repository structure

```text
configs/                 Experiment configuration files
logs/                    Recovered training metrics
scripts/                 Validation and figure-generation scripts
REPRODUCIBILITY.md       Reproduction instructions and known limitations
requirements.txt         Python dependencies
```

Training notebooks, vocabulary mappings, and checkpoints will be added after their paths and loading procedure are validated in a fresh Colab runtime.

## Quick start

```bash
git clone https://github.com/techtrailhead/paatra.git
cd paatra
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Load WikiText-103 with the full Hugging Face repository identifier:

```python
from datasets import load_dataset

corpus = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    split="train",
)
```

Using `load_dataset("wikitext", ...)` can fail with recent Hugging Face Hub versions because the Hub expects a `namespace/name` repository identifier.

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the reported configurations, training settings, expected logs, special-token warning, and supported claims.

## Known limitations

- The completed Config A checkpoint is unavailable.
- Config C stopped after step 14,000.
- The recovered models are not parameter matched.
- The exact seed and complete software freeze still need to be recovered from the original notebooks.
- Token-level perplexity is not directly comparable across different tokenizations.
- Config C inherited BOS/EOS IDs outside its reduced vocabulary and must be corrected before evaluation.

## Citation

Citation metadata will be added after the paper record is finalized.
