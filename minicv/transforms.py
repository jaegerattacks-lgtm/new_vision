import numpy as np
from .utils import validate_image


def resize_nearest(img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """
    Resize image using nearest-neighbor interpolation.

    Parameters
    ----------
    img : np.ndarray
        Grayscale or RGB image.
    new_shape : tuple[int, int]
        (new_h, new_w)
    """
    validate_image(img)

    if not isinstance(new_shape, tuple) or len(new_shape) != 2:
        raise TypeError("new_shape must be a tuple (new_h, new_w).")

    new_h, new_w = new_shape
    if new_h <= 0 or new_w <= 0:
        raise ValueError("new_shape values must be positive.")

    old_h, old_w = img.shape[:2]

    row_idx = np.round(np.linspace(0, old_h - 1, new_h)).astype(int)
    col_idx = np.round(np.linspace(0, old_w - 1, new_w)).astype(int)

    if img.ndim == 2:
        return img[row_idx][:, col_idx].astype(np.float32)
    return img[row_idx][:, col_idx, :].astype(np.float32)


def resize_bilinear(img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """
    Resize image using bilinear interpolation.

    Parameters
    ----------
    img : np.ndarray
        Input image, either grayscale (H, W) or RGB (H, W, 3).
    new_shape : tuple[int, int]
        Target size as (new_height, new_width).

    Returns
    -------
    np.ndarray
        Resized image.

    Raises
    ------
    TypeError
        If new_shape is not a tuple of length 2.
    ValueError
        If new dimensions are not positive.

    Notes
    -----
    Bilinear interpolation computes each output pixel as a weighted
    average of the four nearest pixels in the source image.
    """
    validate_image(img)

    if not isinstance(new_shape, tuple) or len(new_shape) != 2:
        raise TypeError("new_shape must be a tuple (new_h, new_w).")

    new_h, new_w = new_shape
    if new_h <= 0 or new_w <= 0:
        raise ValueError("new_shape values must be positive.")

    old_h, old_w = img.shape[:2]

    if img.ndim == 2:
        channels = 1
        src = img[..., None]
    else:
        channels = 3
        src = img

    out = np.zeros((new_h, new_w, channels), dtype=np.float32)

    x_scale = (old_w - 1) / max(new_w - 1, 1)
    y_scale = (old_h - 1) / max(new_h - 1, 1)

    for y in range(new_h):
        for x in range(new_w):
            src_x = x * x_scale
            src_y = y * y_scale

            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, old_w - 1)
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, old_h - 1)

            dx = src_x - x0
            dy = src_y - y0

            top = (1 - dx) * src[y0, x0] + dx * src[y0, x1]
            bottom = (1 - dx) * src[y1, x0] + dx * src[y1, x1]
            out[y, x] = (1 - dy) * top + dy * bottom

    if img.ndim == 2:
        return out[..., 0]
    return out


def translate_image(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """
    Translate image by tx, ty.
    Positive tx -> right
    Positive ty -> down
    """
    validate_image(img)

    if not isinstance(tx, int) or not isinstance(ty, int):
        raise TypeError("tx and ty must be integers.")

    out = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape[:2]

    src_x0 = max(0, -tx)
    src_x1 = min(w, w - tx)
    src_y0 = max(0, -ty)
    src_y1 = min(h, h - ty)

    dst_x0 = max(0, tx)
    dst_x1 = min(w, w + tx)
    dst_y0 = max(0, ty)
    dst_y1 = min(h, h + ty)

    if img.ndim == 2:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    else:
        out[dst_y0:dst_y1, dst_x0:dst_x1, :] = img[src_y0:src_y1, src_x0:src_x1, :]

    return out


def rotate_image(img: np.ndarray, angle: float, interpolation: str = "nearest") -> np.ndarray:
    """
    Rotate image about center using inverse mapping.

    Parameters
    ----------
    interpolation : str
        'nearest' or 'bilinear'
    """
    validate_image(img)

    if interpolation not in ("nearest", "bilinear"):
        raise ValueError("interpolation must be 'nearest' or 'bilinear'.")

    h, w = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    if img.ndim == 2:
        channels = 1
        src = img[..., None]
    else:
        channels = 3
        src = img

    out = np.zeros((h, w, channels), dtype=np.float32)

# using inverse mapping to avoid holes in the rotated image
    for y_out in range(h):
        for x_out in range(w):
            x_shift = x_out - cx
            y_shift = y_out - cy

            x_src =  cos_a * x_shift + sin_a * y_shift + cx
            y_src = -sin_a * x_shift + cos_a * y_shift + cy

            if 0 <= x_src < w and 0 <= y_src < h:
                if interpolation == "nearest":
                    x_n = int(round(x_src))
                    y_n = int(round(y_src))
                    x_n = min(max(x_n, 0), w - 1)
                    y_n = min(max(y_n, 0), h - 1)
                    out[y_out, x_out] = src[y_n, x_n]
                else:
                    x0 = int(np.floor(x_src))
                    x1 = min(x0 + 1, w - 1)
                    y0 = int(np.floor(y_src))
                    y1 = min(y0 + 1, h - 1)

                    dx = x_src - x0
                    dy = y_src - y0

                    top = (1 - dx) * src[y0, x0] + dx * src[y0, x1]
                    bottom = (1 - dx) * src[y1, x0] + dx * src[y1, x1]
                    out[y_out, x_out] = (1 - dy) * top + dy * bottom

    if img.ndim == 2:
        return out[..., 0]
    return out



#newly added transforms, for more augmentation transforms

def flip_horizontal(img: np.ndarray) -> np.ndarray:
    """Flip image horizontally (left to right)."""
    return img[:, ::-1].copy()

def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness: factor > 1 brightens, < 1 darkens."""
    # Ensure we don't overflow 1.0 (assuming float32 [0,1])
    return np.clip(img * factor, 0.0, 1.0)