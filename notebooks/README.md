# PAATRA notebooks

The notebooks in this directory are sanitized Colab exports: notebook code and Markdown are preserved, while bulky widget metadata and most saved outputs are removed. Reference training metrics are stored separately under `logs/`.

## Supported execution order

1. `01_parameter_audit.ipynb` — audit vocabulary and transformer parameter allocation.
2. `02_teacher_kill_gate.ipynb` — run the teacher-only vocabulary screening diagnostic.
3. `03_scaled_setup.ipynb` — prepare WikiText-103 chunks and vocabulary mappings in Google Drive.
4. Run one or more student notebooks:
   - `04_train_A.ipynb`
   - `05_train_B.ipynb`
   - `06_train_C.ipynb`

The setup notebook uses the current Hugging Face dataset identifier:

```python
load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", ...)
```

## Evidence status

- Config A notebook is included for reproduction, but its completed checkpoint was not recovered.
- Config B is the completed 20K run reported in the revised artifact.
- Config C is the 10K run that reached step 14,000 in the recovered session. The public notebook adds intermediate checkpointing and resets invalid inherited BOS/EOS IDs.

## Evaluation notebooks

The original three-way evaluation and benchmark notebooks are being retained outside the supported pipeline until their checkpoint-loading assumptions and stale A/B/C labels are reconciled. In particular, the recovered checkpoint is a custom PyTorch bundle rather than a Hugging Face `save_pretrained()` directory.

A public evaluation notebook should only be marked supported after it can:

1. reconstruct a student from the stored model configuration;
2. load the `.pt` state dictionary in a fresh runtime;
3. load the exact student-to-teacher vocabulary mapping;
4. evaluate the same raw text for every available checkpoint;
5. report byte- or character-normalized negative log-likelihood.
