"""Fixed H/E separation and first-order Morlet-modulus frontend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .control_bank import (
    CONTROL_CONTRACT_ID,
    CONTROL_GENERATOR_VERSION,
    CONTROL_RNG,
    CONTROL_SEED,
    generate_matched_control_bundle,
)
from .identity import domain_hash
from .morlet import generate_morlet_bundle


class FrontendContractError(ValueError):
    """Fixed-frontend input or protected-precision contract failed."""


@dataclass(frozen=True)
class FrontendOutput:
    feature_h: torch.Tensor
    feature_e: torch.Tensor
    valid_support_mask: torch.Tensor
    fixed_frontend_identity: dict[str, str]


@dataclass(frozen=True)
class _FFTCacheKey:
    canonical_kernel_hash: str
    spatial_execution_kernel_hash: str
    input_dimensions: tuple[int, int]
    fft_grid: tuple[int, int]
    dtype: str
    normalization: str
    shift_convention: str
    crop_convention: str
    backend_name: str
    backend_version: str
    device_class: str
    device_index: int | None
    kernel_data_ptr: int
    kernel_version: int


def _autocast_active() -> bool:
    cpu_enabled = getattr(torch, "is_autocast_cpu_enabled", lambda: False)()
    return bool(torch.is_autocast_enabled() or cpu_enabled)


def _check_precision(device: torch.device) -> None:
    if _autocast_active():
        raise FrontendContractError("autocast must be disabled for the fixed frontend")
    if device.type == "cuda" and (
        torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    ):
        raise FrontendContractError("TF32 must be disabled for the fixed frontend")


def complex_convolve(
    image: torch.Tensor,
    kernels: torch.Tensor,
    *,
    backend: str,
    kernel_fft: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reflect-pad and apply canonical complex kernels as true convolution."""

    if image.ndim != 4 or image.shape[1] != 1 or image.dtype != torch.float32:
        raise FrontendContractError("complex convolution requires float32 [B,1,H,W]")
    if kernels.ndim != 3 or not torch.is_complex(kernels):
        raise FrontendContractError("kernels must have complex shape [K,105,105]")
    if kernels.shape[-2:] != (105, 105) or kernels.dtype != torch.complex64:
        raise FrontendContractError("kernels must be canonical complex64 [K,105,105]")
    if image.device != kernels.device:
        raise FrontendContractError("image and fixed kernels must share one device")
    height, width = image.shape[-2:]
    if height <= 52 or width <= 52:
        raise FrontendContractError("input dimensions must be larger than 52 for reflect padding")
    padded = F.pad(image, (52, 52, 52, 52), mode="reflect")
    if backend == "spatial":
        execution = kernels.flip((-2, -1))[:, None]
        real = F.conv2d(padded, execution.real)
        imaginary = F.conv2d(padded, execution.imag)
        return torch.complex(real, imaginary)
    if backend == "fft":
        fft_shape = (height + 208, width + 208)
        image_fft = torch.fft.fft2(padded.to(torch.complex64), s=fft_shape, norm="backward")
        if kernel_fft is None:
            kernel_fft = torch.fft.fft2(kernels, s=fft_shape, norm="backward")
        elif (
            kernel_fft.shape != (kernels.shape[0], *fft_shape)
            or kernel_fft.dtype != torch.complex64
            or kernel_fft.device != kernels.device
        ):
            raise FrontendContractError("cached kernel spectrum is incompatible")
        full = torch.fft.ifft2(
            image_fft[:, 0, None] * kernel_fft[None], s=fft_shape, norm="backward"
        )
        return full[..., 104 : 104 + height, 104 : 104 + width]
    raise FrontendContractError(f"unsupported convolution backend: {backend}")


def _stable_modulus(response: torch.Tensor) -> torch.Tensor:
    real_abs = response.real.abs()
    imag_abs = response.imag.abs()
    high = torch.maximum(real_abs, imag_abs)
    low = torch.minimum(real_abs, imag_abs)
    safe_high = torch.where(high == 0.0, torch.ones_like(high), high)
    result = high * torch.sqrt(1.0 + torch.square(low / safe_high))
    if result.dtype != torch.float32 or not torch.isfinite(result).all():
        raise FrontendContractError("complex modulus produced a non-finite or non-float32 value")
    return result


class _FixedHEWaveletFrontend(nn.Module):
    """Shared immutable H/E separation and wavelet-modulus execution."""

    def __init__(
        self,
        *,
        backend: str,
        kernel_name: str,
        kernels64: np.ndarray,
        channel_metadata: tuple[tuple[int, int, int, str], ...],
        identity_fields: dict[str, str],
        canonical_kernel_hash: str,
        spatial_execution_hash: str,
    ) -> None:
        super().__init__()
        if backend not in {"fft", "spatial"}:
            raise FrontendContractError("backend must be fft or spatial")
        basis = np.array(
            [
                [0.644211, 0.716556, 0.266844],
                [0.092789, 0.954111, 0.283111],
            ],
            dtype=np.float64,
        )
        pseudoinverse = np.linalg.pinv(basis)
        self.backend = backend
        self._kernel_name = kernel_name
        self.register_buffer("stain_basis", torch.from_numpy(basis), persistent=True)
        self.register_buffer(
            "stain_pseudoinverse", torch.from_numpy(pseudoinverse), persistent=True
        )
        self.register_buffer(kernel_name, torch.from_numpy(kernels64.copy()), persistent=True)
        self.channel_metadata = channel_metadata
        stain_header = {
            "background": "white-maps-to-zero",
            "basis": [
                ["0.644211", "0.716556", "0.266844"],
                ["0.092789", "0.954111", "0.283111"],
            ],
            "basis_order": ["H", "E"],
            "clipping": "nonnegative-after-unmixing",
            "input": "srgb-uint8",
            "normalization": "none",
            "od": "-ln(max(I,1)/255)",
            "output_dtype": "float32",
            "payload_length": 0,
            "pseudoinverse": "moore-penrose-float64",
        }
        self._stain_spec_hash = domain_hash("cg/stain-separation-spec/v1", stain_header)
        self._identity_fields = identity_fields
        self._canonical_kernel_hash = canonical_kernel_hash
        self._spatial_execution_hash = spatial_execution_hash
        self._declared_fixed_identity = self.fixed_state_identity()
        self._fixed_identity_cache_token = self._fixed_state_token()
        self._kernel_fft_cache: torch.Tensor | None = None
        self._kernel_fft_cache_key: _FFTCacheKey | None = None

    @property
    def shared_kernel_reference_count(self) -> int:
        return 1

    @property
    def _kernels(self) -> torch.Tensor:
        return getattr(self, self._kernel_name)

    def _fixed_state_token(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                name,
                value.data_ptr(),
                int(value._version),
                value.device,
                value.dtype,
                tuple(value.shape),
            )
            for name in ("stain_basis", "stain_pseudoinverse", self._kernel_name)
            for value in (getattr(self, name),)
        )

    def _forward_fixed_state_identity(self) -> dict[str, str]:
        token = self._fixed_state_token()
        if token != self._fixed_identity_cache_token:
            self._declared_fixed_identity = self.fixed_state_identity()
            self._fixed_identity_cache_token = token
        return dict(self._declared_fixed_identity)

    def separate_stains(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise FrontendContractError("RGB input must have channel shape [B,3,H,W]")
        if rgb.dtype != torch.uint8:
            raise FrontendContractError("canonical fixed-frontend input must be uint8")
        if rgb.device != self.stain_basis.device:
            raise FrontendContractError("RGB input and fixed frontend must share one device")
        _check_precision(rgb.device)
        values = rgb.permute(0, 2, 3, 1).to(torch.float64)
        normalized = torch.clamp(values, min=1.0) / 255.0
        optical_density = -torch.log(normalized)
        concentrations = torch.clamp(optical_density @ self.stain_pseudoinverse, min=0.0)
        output = concentrations.permute(0, 3, 1, 2).to(torch.float32)
        if not torch.isfinite(output).all():
            raise FrontendContractError("stain separation produced non-finite concentrations")
        return output

    def _valid_mask(
        self, batch: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.zeros((batch, 1, height, width), dtype=torch.bool, device=device)
        if height > 104 and width > 104:
            mask[..., 52 : height - 52, 52 : width - 52] = True
        return mask

    def forward(self, rgb: torch.Tensor) -> FrontendOutput:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise FrontendContractError("RGB input must have channel shape [B,3,H,W]")
        height, width = rgb.shape[-2:]
        if height <= 52 or width <= 52:
            raise FrontendContractError("input dimensions must be larger than 52")
        fixed_identity = self._forward_fixed_state_identity()
        concentrations = self.separate_stains(rgb)
        batch = concentrations.shape[0]
        combined = concentrations.reshape(batch * 2, 1, height, width)
        kernel_fft = None
        if self.backend == "fft":
            fft_shape = (height + 208, width + 208)
            cache_key = _FFTCacheKey(
                canonical_kernel_hash=self._canonical_kernel_hash,
                spatial_execution_kernel_hash=self._spatial_execution_hash,
                input_dimensions=(height, width),
                fft_grid=fft_shape,
                dtype=str(self._kernels.dtype).removeprefix("torch."),
                normalization="backward",
                shift_convention="no-shift",
                crop_convention="offset-104-same-size",
                backend_name="torch.fft.fft2-ifft2",
                backend_version=str(torch.__version__),
                device_class=rgb.device.type,
                device_index=rgb.device.index,
                kernel_data_ptr=self._kernels.data_ptr(),
                kernel_version=int(self._kernels._version),
            )
            if self._kernel_fft_cache is None or self._kernel_fft_cache_key != cache_key:
                self._kernel_fft_cache = torch.fft.fft2(
                    self._kernels,
                    s=fft_shape,
                    norm="backward",
                )
                self._kernel_fft_cache_key = cache_key
            kernel_fft = self._kernel_fft_cache
        response = complex_convolve(
            combined,
            self._kernels,
            backend=self.backend,
            kernel_fft=kernel_fft,
        )
        modulus = _stable_modulus(response).reshape(batch, 2, 4, 8, height, width)
        return FrontendOutput(
            feature_h=modulus[:, 0],
            feature_e=modulus[:, 1],
            valid_support_mask=self._valid_mask(batch, height, width, rgb.device),
            fixed_frontend_identity=fixed_identity,
        )

    def fixed_state_identity(self) -> dict[str, str]:
        digest = hashlib.sha256()
        for name in ("stain_basis", "stain_pseudoinverse", self._kernel_name):
            value = getattr(self, name).detach().cpu().contiguous().numpy()
            digest.update(name.encode("ascii") + b"\x00")
            digest.update(value.tobytes(order="C"))
        return {
            "stain_spec_hash": self._stain_spec_hash,
            **self._identity_fields,
            "fixed_state_sha256": "sha256:" + digest.hexdigest(),
        }


class FixedHEMorletFrontend(_FixedHEWaveletFrontend):
    """Immutable primary Morlet frontend; this module owns no Parameter."""

    def __init__(
        self, *, backend: str, sigma0: str = "0.8", xi0: str = "3*pi/4", gamma: str = "0.5"
    ) -> None:
        bundle = generate_morlet_bundle(sigma0=sigma0, xi0=xi0, gamma=gamma)
        self.morlet_parameters = {"sigma0": sigma0, "xi0": xi0, "gamma": gamma}
        super().__init__(
            backend=backend,
            kernel_name="morlet_kernels",
            kernels64=bundle.kernels64,
            channel_metadata=bundle.channel_metadata,
            identity_fields={
                "morlet_parameter_hash": bundle.parameter_hash,
                "canonical_kernel_hash": bundle.canonical_kernel_hash,
                "spatial_execution_hash": bundle.spatial_execution_hash,
            },
            canonical_kernel_hash=bundle.canonical_kernel_hash,
            spatial_execution_hash=bundle.spatial_execution_hash,
        )
        self._parameter_hash = bundle.parameter_hash

    def artifact_identity(self) -> dict[str, str]:
        parameters = self.morlet_parameters
        changed = parameters != {"sigma0": "0.8", "xi0": "3*pi/4", "gamma": "0.5"}
        return {
            "frontend_variant": "morlet",
            "frontend_contract_id": (
                "fixed-he-morlet-phase2a-linear-v1" if changed else "fixed-he-morlet-linear-v1"
            ),
            **(parameters if changed else {}),
            **self.fixed_state_identity(),
        }


class FixedHEMatchedControlFrontend(_FixedHEWaveletFrontend):
    """Immutable envelope-matched random-phase control frontend."""

    def __init__(self, *, backend: str) -> None:
        bundle = generate_matched_control_bundle()
        super().__init__(
            backend=backend,
            kernel_name="control_kernels",
            kernels64=bundle.kernels64,
            channel_metadata=bundle.channel_metadata,
            identity_fields={
                "filter_bank_specification_hash": bundle.specification_hash,
                "canonical_kernel_hash": bundle.canonical_kernel_hash,
                "spatial_execution_hash": bundle.spatial_execution_hash,
            },
            canonical_kernel_hash=bundle.canonical_kernel_hash,
            spatial_execution_hash=bundle.spatial_execution_hash,
        )

    def artifact_identity(self) -> dict[str, str]:
        return {
            "frontend_variant": "matched_control",
            "frontend_contract_id": CONTROL_CONTRACT_ID,
            "generator_version": CONTROL_GENERATOR_VERSION,
            "rng": CONTROL_RNG,
            "control_seed": str(CONTROL_SEED),
            **self.fixed_state_identity(),
        }
