## What and why

## Data-safety checklist
- [ ] No files under `data/` are staged (other than `README.md` / `checksums.txt`)
- [ ] No patient-level outputs added (`results/per_patient/`, `*.parquet`, `*.npy`, embeddings, etc.)
- [ ] Notebook outputs are stripped
- [ ] Any new committed result is aggregate-only (no per-row patient data)

## If this touches graph construction, censoring, or splits
- [ ] Added/updated a test in `tests/test_censoring.py` or `tests/test_graphs.py`
- [ ] Stated whether previously-generated results are invalidated by this change
