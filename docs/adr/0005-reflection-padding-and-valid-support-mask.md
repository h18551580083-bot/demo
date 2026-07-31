# Use reflection padding and expose boundary influence

The frontend reflects each H/E map by the 52-pixel kernel radius, performs valid
true convolution, and returns a Boolean mask that is `True` only where the complete
receptive field came from the original patch. This avoids the discontinuities of
zero padding and the false adjacency of circular convolution while making the
remaining mirror-context influence explicit; FFT execution must therefore use
zero-padded linear convolution rather than same-size periodization.
