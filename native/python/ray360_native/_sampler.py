"""Versioned sampling interface used by the optional backend dispatcher."""

import numpy as np

from . import _native

# NumPy 1.x promotes float32 coordinate arithmetic for larger dimensions.
_MAX_FLOAT32_SOURCE_DIM = 65_535


def try_sample_v1(image, xy, *, wrap, interpolation, valid):
    """Sample normalized arrays, or return None for the core to handle the call."""
    if (
        interpolation not in ("nearest", "bilinear")
        or image.ndim not in (2, 3)
        or not image.size
        or image.dtype != np.uint8
        or not image.flags.c_contiguous
        or xy.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not xy.flags.c_contiguous
        or not xy.flags.aligned
        or xy.ndim == 0
        or xy.shape[-1] != 2
        or (xy.dtype == np.float32 and max(image.shape[:2]) > _MAX_FLOAT32_SOURCE_DIM)
        or (
            valid is not None
            and (
                type(valid) is not np.ndarray
                or valid.dtype != bool
                or not valid.flags.c_contiguous
                or valid.shape != xy.shape[:-1]
            )
        )
    ):
        return None

    kernel = _native.sample_u8_f32 if xy.dtype == np.float32 else _native.sample_u8_f64
    # NumPy bool arrays can contain noncanonical true bytes. Rust reads raw bytes.
    mask_bytes = None if valid is None else valid.view(np.uint8)
    return kernel(image, xy, mask_bytes, bool(wrap), interpolation == "nearest")


def _sample_numpy(image, xy, *, wrap, interpolation="bilinear", valid):
    try:
        from ray360 import remap
    except ModuleNotFoundError as error:
        if error.name != "ray360":
            raise
        raise ImportError("Install ray360 to sample this input with the NumPy fallback.") from error

    # Released Ray360 versions have a NumPy-only remap.sample.
    reference = getattr(remap, "_sample_numpy", remap.sample)
    return reference(image, xy, wrap=wrap, interpolation=interpolation, valid=valid)


def sample(image, xy, *, wrap=False, interpolation="bilinear", valid=None):
    """Accelerate contiguous uint8 arrays, with NumPy fallback for other inputs."""
    image, xy = np.asarray(image), np.asarray(xy)
    result = try_sample_v1(image, xy, wrap=wrap, interpolation=interpolation, valid=valid)
    if result is not None:
        return result
    return _sample_numpy(image, xy, wrap=wrap, interpolation=interpolation, valid=valid)
