# MD-SOP v2.2 model card

## Intended use and identity

Research prediction of per-residue protein S² from frozen sequence-derived ESM-C/HMM features. These are sequence-input models, but outputs reflect the label definitions, simulation conditions and protein coverage of their supervision sources. They are not direct measurements or a generic measure of all protein motion timescales.

Release experiment: `v2_2_20261107`; training seed 20261007. All six A/B/C × w1/w15 routes are frozen. Inference exports contain identical trained tensors to accepted best checkpoints; packaging makes no scientific parameter change.

## Architecture and training

LayerNorm1182 → Linear128/GELU/dropout0.1 → packed two-layer BiLSTM (256/direction; recurrent dropout0.2) → 512-dimensional residue states → 128-unit regression head. w1 uses linear layers; w15 uses width15/padding7 convolution then a width1 output. Both have GELU/dropout0.2 in the head. Parameter counts: w1=2587069, w15=3504573. No VAE, sigmoid, target scaling or prediction clipping.

Protein-record macro SmoothL1(beta0.05); AdamW lr1e-4/weight decay1e-4. Batch32; C uses16 MD +16 S with equal source losses. A virtual epoch is69 updates. Maximum100 epochs; gradient clipping1; ReduceLROnPlateau(patience3,factor0.5,min_lr1e-6); early stop patience10/min_delta1e-4. Best uses the strict minimum validation criterion, not the patience-reset rule. No resume was used in the six formal runs.

## Supervision and protection

MD train/validation/test:1070/153/136 records. New targets average complete nonoverlapping5τc windows within each of three replicas, then average replicas equally. S-OPPE train/validation/test:2188/273/273 roles, with one zero-supervision training-role record excluded from sampling (2187 supervised training records). Experimental10:10 proteins,982 full-sequence positions,709 scored positions. 26 additional experimental references are protected, not fine-tuned on. 21 reference-related S-OPPE records are quarantined.

Split groups are frozen joint connected components under30% sequence identity and reciprocal80% coverage, not a claim that no conceivable evolutionary relationship remains. Roles and masks are unchanged from v2.1. MD-test contains123 unique sequences/104 groups; S-test269 sequences/206 groups; Experimental10 has8 groups.

## Interpretation and limitations

Primary scores average per-record metrics. PCC/Spearman are averaged only over records where defined; undefined correlations remain missing, not zero. Bias=prediction−target. Pooled/unique-sequence/group summaries are secondary and need not rank models identically.

MD-full100ns is auxiliary; main MD comparison uses the same5τc target for both model versions. Changing targets alone changes the score even for identical B predictions. A lower diagonal score is not an isolated training gain. Window estimators and tail coverage also differ; this is not a pure timescale intervention or proof of greater physical truth.

Only one training seed; group bootstrap does not measure optimization-seed stability. All holdouts were historically reused. Experimental comparison uses8 group blocks; the minimum two-sided exact sign-flip P is2/256, so a24-test Holm family cannot reject at0.05. Nonsignificance does not demonstrate equivalence; unadjusted confidence intervals cannot establish familywise superiority.

Historical original SOPPCL has known overlap with1/10 exact experimental sequences and5/10 joint homology groups. Other historical pipelines differ in training/features/splits; do not assume the same overlap count for them. Their experimental aggregate scores are context, not a matched label-only intervention. Mixed supervision and w15 are not uniformly better; no universal deployment winner is declared.

Prediction outside training length/composition/family coverage, intrinsically disordered regions, membrane/multidomain proteins, altered simulation conditions or experimental measurement definitions requires independent validation. This release has no clinical or safety-critical validation claim.
