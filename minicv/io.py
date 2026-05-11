import numpy as np
import matplotlib.pyplot as plt


def read_image(path: str) -> np.ndarray:
    """
    Read an image from disk and return float32 NumPy array in range [0, 1].

    Parameters
    ----------
    path : str
        Path to image file.

    Returns
    -------
    np.ndarray
        Image array, grayscale (H, W) or RGB(A) converted to RGB (H, W, 3).

    Raises
    ------
    TypeError
        If path is not a string.
    ValueError
        If image cannot be loaded.
    """
    if not isinstance(path, str):
        raise TypeError("path must be a string.")

    try:
        img = plt.imread(path)
    except Exception as e:
        raise ValueError(f"Failed to read image from '{path}': {e}")

    img = np.asarray(img)

    if img.ndim not in (2, 3):
        raise ValueError("Loaded image must be 2D grayscale or 3D RGB/RGBA.")

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)   #return normalize 


def save_image(img: np.ndarray, path: str) -> None:
    """
    Save image array to disk.

    Parameters
    ----------
    img : np.ndarray
        Grayscale or RGB image in any numeric range.
    path : str
        Output image path.

    Raises
    ------
    TypeError
        If img is not ndarray or path is not string.
    ValueError
        If image shape is invalid.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array.")
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if img.ndim not in (2, 3):
        raise ValueError("img must be grayscale (H,W) or RGB (H,W,3).")
    if img.ndim == 3 and img.shape[2] != 3:
        raise ValueError("RGB image must have shape (H, W, 3).")

    plt.imsave(path, np.clip(img, 0.0, 1.0), cmap="gray" if img.ndim == 2 else None)


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.

    Parameters
    ----------
    img : np.ndarray
        RGB image or grayscale image.

    Returns
    -------
    np.ndarray
        Grayscale image.

    Raises
    ------
    TypeError
        If input is not ndarray.
    ValueError
        If input shape is invalid.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array.")

    if img.ndim == 2:
        return img.astype(np.float32)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be grayscale (H,W) or RGB (H,W,3).")

    gray = np.dot(img[..., :3], np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
    return gray.astype(np.float32)


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to RGB by repeating channels.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image of shape (H, W).

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array.")
    if img.ndim != 2:
        raise ValueError("gray_to_rgb expects a grayscale image of shape (H, W).")

    return np.stack([img, img, img], axis=-1).astype(np.float32)
