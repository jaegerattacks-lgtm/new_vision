import numpy as np
from .utils import validate_image, validate_grayscale, validate_kernel, pad_image


def convolve2d(img: np.ndarray, kernel: np.ndarray, pad_mode: str = "constant") -> np.ndarray:
    """
    Perform true 2D convolution on a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).
    kernel : np.ndarray
        2D convolution kernel with odd dimensions.
    pad_mode : str, optional
        Padding mode used at image boundaries.
        Supported modes: "constant", "edge", "reflect".

    Returns
    -------
    np.ndarray
        Convolved grayscale image with the same shape as input.

    Raises
    ------
    TypeError
        If img or kernel is not a NumPy array.
    ValueError
        If img is not grayscale, kernel is empty, not 2D, or has even dimensions.

    Notes
    -----
    True convolution is implemented by flipping the kernel horizontally
    and vertically before applying the sliding window operation.
    """
    validate_grayscale(img)
    validate_kernel(kernel)

    kernel_flipped = np.flipud(np.fliplr(kernel)).astype(np.float32)
    k_h, k_w = kernel_flipped.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded = pad_image(img, max(pad_h, pad_w), mode=pad_mode)

    if pad_h != pad_w:
        start_r = max(pad_h, pad_w) - pad_h
        start_c = max(pad_h, pad_w) - pad_w
        padded = padded[start_r:start_r + img.shape[0] + 2 * pad_h,
                        start_c:start_c + img.shape[1] + 2 * pad_w]

    out = np.zeros_like(img, dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i + k_h, j:j + k_w]
            out[i, j] = np.sum(window * kernel_flipped)

    return out


def filter2d(img: np.ndarray, kernel: np.ndarray, pad_mode: str = "constant") -> np.ndarray:
    """
    Apply 2D convolution-based filtering to grayscale or RGB image.
    RGB is filtered per channel.
    """
    validate_image(img)
    validate_kernel(kernel)

    if img.ndim == 2:
        return convolve2d(img, kernel, pad_mode=pad_mode)

    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        out[..., c] = convolve2d(img[..., c], kernel, pad_mode=pad_mode)
    return out


def mean_filter(img: np.ndarray, kernel_size: int = 3, pad_mode: str = "reflect") -> np.ndarray:
    """
    Apply mean/box filter.
    """
    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return filter2d(img, kernel, pad_mode=pad_mode)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate normalized 2D Gaussian kernel.
    """
    if not isinstance(size, int):
        raise TypeError("size must be an integer.")
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_filter(img: np.ndarray, size: int = 5, sigma: float = 1.0, pad_mode: str = "reflect") -> np.ndarray:
    """
    Apply Gaussian filter to an image.

    Parameters
    ----------
    img : np.ndarray
        Input image, either grayscale (H, W) or RGB (H, W, 3).
    size : int, optional
        Gaussian kernel size. Must be a positive odd integer.
    sigma : float, optional
        Standard deviation of the Gaussian distribution.
    pad_mode : str, optional
        Padding mode used at image boundaries.
        Supported modes: "constant", "edge", "reflect".

    Returns
    -------
    np.ndarray
        Smoothed image with the same shape as input.

    Raises
    ------
    TypeError
        If size is not an integer.
    ValueError
        If size is not a positive odd integer or sigma <= 0.

    Notes
    -----
    The Gaussian kernel is generated first, then applied through
    the general convolution-based filtering pipeline.
    """
    kernel = gaussian_kernel(size, sigma)
    return filter2d(img, kernel, pad_mode=pad_mode)


def median_filter(img: np.ndarray, kernel_size: int = 3, pad_mode: str = "reflect") -> np.ndarray:
    """
    Apply median filter.
    Looping is justified because median requires neighborhood sorting/selection.
    """
    validate_image(img)

    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    pad = kernel_size // 2
    padded = pad_image(img, pad, mode=pad_mode)
    out = np.zeros_like(img, dtype=np.float32)

    if img.ndim == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                out[i, j] = np.median(window)
        return out

    for c in range(3):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                window = padded[i:i + kernel_size, j:j + kernel_size, c]
                out[i, j, c] = np.median(window)
    return out


def global_threshold(img: np.ndarray, thresh_value: float) -> np.ndarray:
    """
    Apply global thresholding to grayscale image.
    """
    validate_grayscale(img)
    if not (0.0 <= thresh_value <= 1.0):
        raise ValueError("thresh_value must be in range [0, 1].")
    return (img > thresh_value).astype(np.float32)


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """
    Compute Otsu optimal threshold and return binary image.
    """
    validate_grayscale(img)

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    total = img_u8.size

    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    weight_bg = 0.0
    max_var = -1.0
    best_thresh = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > max_var:
            max_var = between_var
            best_thresh = t

    threshold = best_thresh / 255.0
    return (img > threshold).astype(np.float32)


def adaptive_threshold_mean(img: np.ndarray, block_size: int = 11, c: float = 0.02) -> np.ndarray:
    """
    Adaptive mean thresholding.
    Each pixel is compared against local mean minus c.
    """
    validate_grayscale(img)

    if not isinstance(block_size, int):
        raise TypeError("block_size must be an integer.")
    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer > 1.")

    local_mean = mean_filter(img, kernel_size=block_size, pad_mode="reflect")
    return (img > (local_mean - c)).astype(np.float32)


def sobel_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Sobel gradients gx and gy.
    """
    validate_grayscale(img)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    gx = convolve2d(img, kx, pad_mode="reflect")
    gy = convolve2d(img, ky, pad_mode="reflect")
    return gx, gy


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Compute normalized Sobel edge magnitude.
    """
    gx, gy = sobel_gradients(img)
    mag = np.sqrt(gx**2 + gy**2)
    m = mag.max()
    if m == 0:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / m).astype(np.float32)


def bit_plane_slice(img: np.ndarray, plane: int) -> np.ndarray:
    """
    Extract bit plane from grayscale image.
    plane must be in [0, 7].
    """
    validate_grayscale(img)

    if not isinstance(plane, int):
        raise TypeError("plane must be an integer.")
    if plane < 0 or plane > 7:
        raise ValueError("plane must be between 0 and 7.")

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return (((img_u8 >> plane) & 1).astype(np.float32))


def laplacian_filter(img: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """
    Additional technique 1: Laplacian filter.
    """
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    return filter2d(img, kernel, pad_mode=pad_mode)


def log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Additional technique 2: Log transformation.
    """
    validate_image(img)
    if c <= 0:
        raise ValueError("c must be > 0.")
    out = c * np.log1p(np.clip(img, 0.0, None))
    mx = out.max()
    if mx == 0:
        return np.zeros_like(out, dtype=np.float32)
    return (out / mx).astype(np.float32)
