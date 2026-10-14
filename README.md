# DREAMT sleep-stage classification — Version 6

[Open the notebook in Google Colab](https://colab.research.google.com/github/abdulazizabid/GPT/blob/main/DREAMT_Version_6.ipynb)

Version 6 investigates the student model's zero REM F1 in Version 5. It adds class-balanced training, a separate REM-focused candidate, longer temporal context, validation-only decision tuning, compatible cache reuse and resumable GPU-aware training.

- [Notebook](DREAMT_Version_6.ipynb)
- [Cell-by-cell code](DREAMT_Version_6_cells.md)
- [Diagnosis and changes](REM_Diagnosis_and_V6_Changes.md)
- [Validation report](validation_report.json)

## Run

Open the notebook in Colab, choose a GPU runtime when available, set `DATA`, `V5` and `OUT` in the Configuration cell, and run all cells in order. Keep the V6 output directory separate from V5. The DREAMT raw CSV files, V5 caches and trained weights are not included in this repository.

The notebook reuses compatible preprocessing artifacts; otherwise it recomputes preprocessing and trains new student models. CUDA mixed precision is automatic, with CPU fallback.

## Validation status

Notebook structure, synthetic CPU end-to-end execution, cache reuse and interrupted-training resume passed. Actual DREAMT score improvements and CUDA execution have not been measured here. Higher REM F1, macro F1 and accuracy must be established by a real-data training and evaluation run; they are not guaranteed.
