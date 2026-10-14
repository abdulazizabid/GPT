# DREAMT V7: stronger wearable inputs and genuine knowledge distillation

The targets are macro F1 of at least 70% and accuracy around 70%. They are goals to evaluate after training, not guaranteed results. V6 improved REM F1 from zero to 21.97% and macro F1 to 53.91%, but its adjusted accuracy fell to 67.91%. Its saved experiment contained no teacher targets or KD run.

V7 changes the student from IBI-only to the user-approved multimodal wearable specification: BVP, derived pulse intervals, acceleration, EDA, temperature and heart rate. PSG is available only to the teachers. A multimodal score must be described as such when comparing with V5/V6.

## Main fixes

1. **Use native 64 Hz wearables throughout development and testing.** The 100 Hz CSVs supply training-only PSG. This avoids relying on synthetic 64 Hz training views while evaluating native 64 Hz recordings, and applies one label interpretation to all splits.
2. **Separate scoring labels from context.** Retain complete wearable epochs when labels are missing, preparation, or ambiguous. Mask those labels from supervision and metrics. Collapse NREM substages before checking three-class purity. True missing time still breaks sequences. Do not invent preparation boundaries or discard genuine Wake.
3. **Audit paired sources.** Join PSG and wearables by subject and absolute epoch index. Teacher supervision requires valid signals and agreeing three-class annotations. Strong source disagreement produces a useful error rather than an automatic time shift. Valid unscored PSG remains teacher context; PSG amplitude, including chin EMG amplitude, is preserved before train-only normalization.
4. **Actually train the teachers.** Three cross-fitted three-class teachers generate OOF targets for student-training participants. Each held-out participant is excluded from teacher fitting, normalization and checkpoint selection. An inner stopping subset is drawn from the other training people. Missing compatible teacher checkpoints/targets are trained or reconstructed automatically.
5. **Distill with hard-label safeguards.** Train a supervised multimodal reference, then initialize two distilled students from it. Use moderate class-balanced cross-entropy, a sleep-only REM auxiliary objective, and delayed KD. Use calibrated three-class teacher logits and class-balanced, teacher-correct KD gates. Selected distilled checkpoints must have received KD updates.
6. **Select without test tuning.** Compare the two distilled students and their equal-probability ensemble using the eight model-selection participants. Apply an accuracy guard, with an explicit reported fallback if it fails. The other eight validation participants fit temperature only. Class offsets are disabled. Report whether the chosen pipeline is a single model or an ensemble.
7. **Measure comparable outcomes.** Report supervised and distilled scores, class confusion, per-person results, coverage and subject-bootstrap intervals. When previous predictions exist, compare V5/V6 and V7 on matched epochs. Target attainment is stated for both the new scored cohort and available matched historical cohorts.

The supervised reference is not a matched-budget causal KD ablation: the distilled models also receive additional fine-tuning and may be ensembled. The comparison measures the complete resulting pipelines; it does not attribute every gain solely to KD.

## Run

Open `DREAMT_Version_7.ipynb` in Colab. Select a GPU when available, then edit `DATA`, `V5`, `V6`, and `OUT` in Configuration. Keep `OUT` separate from the earlier versions. Run all cells in order. Expect a one-time preprocessing rebuild and real teacher training, which can take hours. The final cell collects a review ZIP without raw CSVs or model weights.

The old caches omit the new sensor inputs and rejected context, so importing them as complete V7 data would be incorrect. Earlier directories remain read only. Compatible V7 prepared arrays, teacher folds, OOF targets, checkpoints, calibration and predictions are reused. Missing artifacts are recalculated; interrupted stages resume. Old predictions are optional historical comparison inputs, not training labels.

## GPU and storage

The code uses one CUDA GPU when available, mixed precision, TF32 where supported, clipped gradients, effective-batch accumulation, pinned transfers, local SSD mirrors and memory-mapped prepared arrays. CPU fallback works but is slower. CPU signal preprocessing remains necessary. Prepared PSG and local mirrors need several GB of free storage; the notebook reports available disk/GPU memory. Lower `BATCH` after an OOM and rerun to resume from the last completed saved optimizer update. Smaller microbatches preserve sample coverage/effective window batches but need not produce numerically identical gradients.

## Evidence and limits

The published validation report covers synthetic CPU checks and records the exact notebook hash. Real DREAMT training and CUDA execution cannot be performed on the current host because the dataset/weights and GPU are unavailable. A 70% result must come from the generated real-data metrics; a successful install, synthetic test or KD gradient check is not evidence of that score. The historical test has already been inspected; another independent participant cohort provides stronger final confirmation.
