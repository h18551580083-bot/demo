# Use complex Morlet wavelets in the primary fixed frontend

The primary shared H/E wavelet backbone uses only zero-DC two-dimensional complex
Morlet wavelets represented by paired real and imaginary kernels. This preserves a
clear scale-orientation channel model aligned with scattering literature and
trades the lower channel count of real or mixed filter families for quadrature
responses with cleaner phase handling; LoG and real Gabor filters remain possible
future ablations rather than primary channels.
