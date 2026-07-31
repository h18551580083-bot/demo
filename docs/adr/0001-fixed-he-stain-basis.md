# Use a fixed Ruifrok–Johnston H/E stain basis

The baseline uses the ordered Ruifrok–Johnston H and E RGB absorbance vectors with
an explicit two-vector pseudoinverse, rather than stain normalization, per-image
estimation, or a library HED default. This deliberately trades adaptation to stain
variation for a data-independent, hashable frontend whose meaning remains stable
across CAM16 development and later transfer evaluation. RGB is converted to
optical density using a locked white-reference convention; negative unmixed
concentrations are clipped to zero and no data-fitted scaling is applied.
