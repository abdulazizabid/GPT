# DREAMT V5 diagnosis and V6 changes

The supplied V5 results show 72.05% accuracy, 47.22% macro F1, and 0% REM F1 on 13,829 retained epochs, including 1,693 true REM epochs. The precision/recall figures imply approximately 2,502 Wake predictions and 11,327 NREM predictions, accounting for the full test set. The likely behavior is therefore a student that never selects REM.

## What is supported by the code

- The label mapping is consistent: REM is five-class label 4 and student label 2. There is no evident REM index bug.
- The student uses ordinary, unweighted cross-entropy. Its knowledge-distillation and consistency losses are active from the first epoch. This can favor the majority solution, but training supports and teacher performance must be measured before confirming the cause.
- The teacher's three NREM probabilities are summed correctly. Weak REM teacher targets could nevertheless reinforce the student's collapse. The uploaded notebook does not show the OOF teacher's three-class REM performance.
- Student validation macro F1 peaks around 0.398 at epoch 6, and training stops at epoch 11. This is a weak student before final testing; the problem is not solely probability calibration.
- Dividing logits by a positive scalar temperature leaves the predicted class unchanged. V5's temperature scaling cannot restore an unpredicted REM class.
- V5 rejects many epochs. Its mixed/preparation/missing-label exclusions need coverage review; the output alone does not establish a timestamp alignment defect.

## What V6 changes

1. **Moderate class balancing.** Inverse-square-root class weights are computed from training subjects only and capped. They are not combined with class oversampling.
2. **A controlled REM experiment.** A weighted-CE student and a second student with an auxiliary REM objective are trained separately. Selection subjects determine which candidate is used. V5 student weights are not reused in the changed architecture.
3. **A stronger IBI representation.** The student preserves absolute IBI, adds a relative IBI channel and interpretable interpolated-series features, uses GroupNorm, and receives 32 minutes of temporal context. No new sensor modality is introduced. These are potential improvements, not established causal fixes.
4. **Better rate matching.** Most training steps use the simulated 64 Hz view, matching final inference. A small consistency penalty begins only after supervised learning has started.
5. **Separated decisions.** Eight of the original validation subjects select models; eight other validation subjects fit calibration and class offsets. The offset search includes an accuracy constraint and zero-offset fallback. Both raw and adjusted test scores are reported.
6. **Optional, audited KD.** KD defaults to off. If enabled, V5 OOF target reuse requires compatible preprocessing and subject-exclusion manifests. The KD candidate is delayed, low-weight, teacher-correct gated, and skipped when teacher REM recall is too weak. Missing teachers do not prevent supervised training.
7. **Reusable artifacts.** Compatible V5 wearable caches are imported read only. Otherwise raw CSV preprocessing runs again. V6 saves source/config identities, atomic checkpoints, optimizer/scheduler/scaler/RNG state, calibration, per-subject predictions and complete metrics. Invalid data caches are preserved and rebuilt. Changed code or settings require a new run tag/output directory.
8. **Efficient GPU execution.** CUDA is detected automatically; training uses suitable AMP precision, TF32 where supported, conservative memory-aware batches, pinned transfers and RAM-staged wearable arrays. CPU fallback remains available. The notebook uses one GPU.

## How to run and judge the result

Open `DREAMT_Version_6.ipynb`, set `DATA`, `V5`, and `OUT` in Configuration, and run all cells. Keep `OUT` separate from Version-5. Default cohort rules preserve V5 comparability; an optional wearable-only cohort is a separate experiment.

Inspect REM precision, recall, F1, prediction counts and average precision, alongside accuracy and macro F1. When saved V5 predictions exist, the notebook compares the versions on matched subject/epoch pairs. It also saves per-subject metrics, confusion matrices, retained coverage and subject-bootstrap intervals. A higher REM recall does not by itself establish better REM F1 or accuracy.

**Validation limit:** notebook structure and synthetic CPU execution can be checked here. The uploaded files do not include the raw DREAMT data, saved caches or trained weights, and this host has no CUDA device. Real-data improvements and GPU execution therefore remain unmeasured. The already-inspected V5 test set is a historical benchmark; a new independent subject cohort would provide stronger confirmation.
