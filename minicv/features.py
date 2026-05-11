import numpy as np
from .utils import validate_image, validate_grayscale
from .filtering import sobel_gradients


def compute_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute histogram of grayscale image.
    """
    validate_grayscale(img)

    if not isinstance(bins, int):
        raise TypeError("bins must be an integer.")
    if bins <= 0:
        raise ValueError("bins must be > 0.")

    hist, _ = np.histogram(img.ravel(), bins=bins, range=(0.0, 1.0))
    return hist.astype(np.int64)


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """
    Histogram equalization for grayscale image.
    """
    validate_grayscale(img)

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(img_u8.ravel(), minlength=256)
    cdf = hist.cumsum()

    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0:
        return np.zeros_like(img, dtype=np.float32)

    cdf_min = nonzero[0]
    total = img_u8.size

    lut = (cdf - cdf_min) / (total - cdf_min + 1e-8)
    lut = np.clip(lut, 0.0, 1.0)

    return lut[img_u8].astype(np.float32)


def global_feature_vector(img: np.ndarray) -> np.ndarray:
    """
    Global descriptor 1:
    [mean, std, min, max]
    """
    validate_image(img)
    x = img.astype(np.float32)
    return np.array([x.mean(), x.std(), x.min(), x.max()], dtype=np.float32)


def color_moments(img: np.ndarray) -> np.ndarray:
    """
    Global descriptor 2:
    color moments for each channel if RGB,
    otherwise moments for grayscale.
    """
    validate_image(img)

    def moments(channel):
        mean = np.mean(channel)
        variance = np.var(channel)
        skewness = np.mean((channel - mean) ** 3) / (np.std(channel) ** 3 + 1e-8)
        return [mean, variance, skewness]

    if img.ndim == 2:
        return np.array(moments(img), dtype=np.float32)

    feats = []
    for c in range(3):
        feats.extend(moments(img[..., c]))
    return np.array(feats, dtype=np.float32)


def gradient_features(img: np.ndarray) -> np.ndarray:
    """
    Gradient descriptor 1:
    [mean_gradient_magnitude, max_gradient_magnitude]
    """
    validate_grayscale(img)
    gx, gy = sobel_gradients(img)
    mag = np.sqrt(gx**2 + gy**2)
    return np.array([mag.mean(), mag.max()], dtype=np.float32)


def gradient_descriptor(img: np.ndarray) -> np.ndarray:
    """
    Gradient descriptor 2:
    [mean_abs_gx, mean_abs_gy, mean_magnitude, std_magnitude]
    """
    validate_grayscale(img)
    gx, gy = sobel_gradients(img)
    mag = np.sqrt(gx**2 + gy**2)
    return np.array([
        np.mean(np.abs(gx)),
        np.mean(np.abs(gy)),
        np.mean(mag),
        np.std(mag),
    ], dtype=np.float32)
