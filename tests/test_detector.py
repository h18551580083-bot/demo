import pytest

from cam16_wavelet.contracts import DetectorConfig
from cam16_wavelet.optics.detector import SquareLawDetector


@pytest.mark.parametrize(
    "config",
    [
        DetectorConfig(epsilon=0),
        DetectorConfig(noise_std=-1),
        DetectorConfig(quantization_bits=0),
        DetectorConfig(saturation=0),
        DetectorConfig(pool_size=0),
    ],
)
def test_detector_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        SquareLawDetector(config)
