# Keep the primary frontend at first-order wavelet modulus

The primary shared H/E frontend performs one complex Morlet convolution followed
by a modulus and does not cascade a second wavelet transform. This preserves
spatially aligned scale-orientation maps for explicit H/E interaction and avoids
the channel growth of second-order paths, at the cost of not claiming the full
higher-order statistics of a scattering network; the canonical name is therefore
“first-order fixed wavelet-modulus frontend.”
