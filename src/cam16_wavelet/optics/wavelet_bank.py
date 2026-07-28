from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cam16_wavelet.contracts import KernelSpec
from cam16_wavelet.hashing import array_hash


@dataclass(frozen=True)
class KernelBundle:
    kernels: torch.Tensor
    channel_names: tuple[str, ...]
    kernel_hash: str


class WaveletBank:
    @staticmethod
    def build(spec: KernelSpec) -> KernelBundle:
        spec.validate()
        axis = np.arange(spec.size, dtype=np.float64) - spec.size // 2
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        kernels: list[np.ndarray] = []
        names: list[str] = []
        for sigma in spec.scales:
            if "log" in spec.families:
                radius2 = xx**2 + yy**2
                kernel = (radius2 - 2 * sigma**2) * np.exp(-radius2 / (2 * sigma**2))
                kernels.append(kernel)
                names.append(f"log_s{sigma:g}")
            if "gabor" in spec.families:
                for index in range(spec.orientations):
                    theta = np.pi * index / spec.orientations
                    xr = xx * np.cos(theta) + yy * np.sin(theta)
                    yr = -xx * np.sin(theta) + yy * np.cos(theta)
                    envelope = np.exp(-(xr**2 + spec.gamma**2 * yr**2) / (2 * sigma**2))
                    carrier = np.cos(2 * np.pi * xr / (spec.wavelength_factor * sigma))
                    kernels.append(envelope * carrier)
                    names.append(f"gabor_s{sigma:g}_o{index}")
        bank = np.stack(kernels)
        if spec.zero_mean:
            bank -= bank.mean(axis=(-2, -1), keepdims=True)
        if spec.normalization == "l1":
            denominator = np.abs(bank).sum(axis=(-2, -1), keepdims=True)
        else:
            denominator = np.sqrt(np.square(bank).sum(axis=(-2, -1), keepdims=True))
        bank /= np.maximum(denominator, np.finfo(bank.dtype).eps)
        bank32 = bank.astype(np.float32)
        return KernelBundle(torch.from_numpy(bank32), tuple(names), array_hash(bank32))

