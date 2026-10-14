# DREAMT wearable sleep-stage classification

## Version 7 — multimodal student with PSG knowledge distillation

**[Open Version 7 in Google Colab](https://colab.research.google.com/github/abdulazizabid/GPT/blob/main/DREAMT_Version_7.ipynb)**

V7 pursues macro F1 of at least 70% and accuracy around 70%. These are targets to measure, not promised results. It uses the approved wearable signals: BVP, derived pulse intervals, acceleration, EDA, temperature and heart rate. PSG remains teacher-only. This is a broader student input specification than V5/V6's IBI-only model.

The pipeline preserves unscored sensor epochs as temporal context, trains on native 64 Hz wearables for every split, audits PSG/wearable alignment, and trains or resumes three cross-fitted three-class PSG teachers when targets are missing. It trains a supervised reference and two distilled students, selects a single distilled model or ensemble on validation participants, and calibrates temperature without class offsets.

- [V7 notebook](DREAMT_Version_7.ipynb)
- [V7 cell-by-cell code](DREAMT_Version_7_cells.md)
- [Changes and run guide](V7_Changes_and_Run_Guide.md)
- [V7 validation report](V7_validation_report.json)

### Run

Select a Colab GPU when available, set `DATA`, `V5`, `V6`, and `OUT` in Configuration, then run all cells. Keep the V7 output directory separate from previous versions. The dataset, caches and trained weights are not included in this repository.

The new modalities and context policy require a one-time preparation rebuild. Compatible V7 arrays, teachers, targets, models, calibration and predictions are reused; missing items are recalculated or retrained. Expect real teacher training to take hours. CUDA mixed precision, effective-batch accumulation, memory-mapped arrays and local SSD mirrors are automatic; CPU fallback is supported. The final notebook cell creates a result ZIP for review.

### Validation status

Synthetic CPU preprocessing, actual teacher/student forward and backward execution, genuine KD gradients, full pipeline execution, interrupted-update resume, OOF exclusion, deleted-target reconstruction, and completed-cache reuse passed. Actual DREAMT performance and CUDA execution remain unmeasured on this host. Inspect `target_attainment.json` after a real run, and compare both full-cohort and matched historical metrics. No 70% score is claimed here.

## Previous version

- [Version 6 notebook](DREAMT_Version_6.ipynb)
- [Version 6 cell-by-cell code](DREAMT_Version_6_cells.md)
- [Initial REM diagnosis](REM_Diagnosis_and_V6_Changes.md)
- [Version 6 synthetic validation](validation_report.json)
