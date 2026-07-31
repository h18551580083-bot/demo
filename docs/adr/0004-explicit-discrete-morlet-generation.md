# Generate Morlet kernels explicitly with discrete zero-DC projection

The project freezes the Morlet formula and standard scale-frequency parameters but
generates each finite complex kernel itself in `float64`/`complex128`, using a
support- and orientation-specific discrete zero-DC projection rather than relying
only on the continuous correction or Kymatio defaults. This adds generator and
spectral validation obligations, but makes the exact finite kernels independently
hashable, keeps Kymatio out of the runtime contract, and guarantees that H and E
share one auditable kernel tensor.
