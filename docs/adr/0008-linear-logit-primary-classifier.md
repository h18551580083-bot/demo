---
status: accepted
---

# Use an affine logit as the primary classifier

The primary classifier is one explicitly zero-initialized affine map from the
canonical 9408 pooled coordinates to one raw binary logit. This deliberately
chooses auditability and an exact 9409-parameter head over hidden nonlinear
capacity, so the complete electronic backend has exactly 9473 trainable scalars
including the 64 approved cross-stain gates. Richer heads, internal normalization,
and probability or calibration logic are not part of the primary classifier.
