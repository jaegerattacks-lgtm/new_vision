import numpy as np


def validate_image(img: np.ndarray) -> None:
    """
    Validate grayscale or RGB image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array.")
    if img.ndim not in (2, 3):
        raise ValueError("img must be 2D grayscale or 3D RGB.")
    if img.ndim == 3 and img.shape[2] != 3:
        raise ValueError("RGB image must have shape (H, W, 3).")


def validate_grayscale(img: np.ndarray) -> None:
    """
    Validate grayscale image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array.")
    if img.ndim != 2:
        raise ValueError("This function expects a grayscale image of shape (H, W).")


def validate_kernel(kernel: np.ndarray) -> None:
    """
    Validate 2D convolution kernel.
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError("kernel must be a NumPy array.")
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2D.")
    if kernel.size == 0:
        raise ValueError("kernel must be non-empty.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd.")
    if not np.issubdtype(kernel.dtype, np.number):
        raise TypeError("kernel must contain numeric values.")


def clip_image(img: np.ndarray, min_value: float = 0.0, max_value: float = 1.0) -> np.ndarray:
    """
    Clip image pixels to a defined range.
    """
    validate_image(img)
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value.")
    return np.clip(img, min_value, max_value).astype(np.float32)


def normalize_image(img: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """
    Normalize image with 3 modes:
    - minmax   -> [0,1]
    - mean     -> centered by mean and scaled by range
    - standard -> z-score
    """
    validate_image(img)
    x = img.astype(np.float32) 

    if mode == "minmax":
        mn, mx = x.min(), x.max()
        if mx == mn:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mn) / (mx - mn)).astype(np.float32)

    if mode == "mean":
        mn, mx = x.min(), x.max()
        if mx == mn:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - x.mean()) / (mx - mn)).astype(np.float32)

    if mode == "standard":
        std = x.std()
        if std == 0:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - x.mean()) / std).astype(np.float32)

    raise ValueError("mode must be one of: 'minmax', 'mean', 'standard'.")


def pad_image(img: np.ndarray, pad_size: int, mode: str = "constant", constant_values: float = 0.0) -> np.ndarray:
    """
    Pad image using at least 3 modes:
    - constant
    - edge
    - reflect
    """
    validate_image(img)

    if not isinstance(pad_size, int):
        raise TypeError("pad_size must be an integer.")
    if pad_size < 0:
        raise ValueError("pad_size must be >= 0.")
    if mode not in ("constant", "edge", "reflect"):
        raise ValueError("mode must be 'constant', 'edge', or 'reflect'.")

    if img.ndim == 2:
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    else:
        pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))

    if mode == "constant":
        return np.pad(img, pad_width, mode=mode, constant_values=constant_values)
    return np.pad(img, pad_width, mode=mode)
