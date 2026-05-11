import io
import numpy as np
import matplotlib.pyplot as plt
from .utils import validate_image


def _prepare_color(img: np.ndarray, color):
    if img.ndim == 2:
        if isinstance(color, (int, float, np.integer, np.floating)):
            return float(color)
        raise ValueError("For grayscale image, color must be a scalar.")
    else:
        if isinstance(color, np.ndarray):
            color = color.tolist()

        if isinstance(color, (tuple, list)) and len(color) == 3:
            return np.array(color, dtype=np.float32)

        raise ValueError("For RGB image, color must be a tuple/list of length 3.")


def draw_point(img: np.ndarray, point: tuple[int, int], color, thickness: int = 1) -> np.ndarray:
    """
    Draw point centered at (x, y).
    """
    validate_image(img)
    if not isinstance(thickness, int) or thickness <= 0:
        raise ValueError("thickness must be a positive integer.")

    out = img.copy().astype(np.float32)
    color = _prepare_color(out, color)

    x, y = point
    r = thickness // 2

    for yy in range(y - r, y + r + 1):
        for xx in range(x - r, x + r + 1):
            if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]:
                out[yy, xx] = color
    return out


def draw_line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color, thickness: int = 1) -> np.ndarray:
    """
    Draw a line on an image using Bresenham's line algorithm.

    Parameters
    ----------
    img : np.ndarray
        Input image, either grayscale (H, W) or RGB (H, W, 3).
    p1 : tuple[int, int]
        Starting point as (x, y).
    p2 : tuple[int, int]
        Ending point as (x, y).
    color : int, float, tuple, or list
        Drawing color. Use scalar for grayscale image and 3-element tuple/list for RGB image.
    thickness : int, optional
        Line thickness in pixels.

    Returns
    -------
    np.ndarray
        Output image with the drawn line.

    Raises
    ------
    ValueError
        If thickness is not a positive integer or color format is invalid.

    Notes
    -----
    The line is generated point by point using Bresenham's algorithm,
    then each point is drawn with the requested thickness.
    """
    validate_image(img)
    if not isinstance(thickness, int) or thickness <= 0:
        raise ValueError("thickness must be a positive integer.")

    out = img.copy().astype(np.float32)
    color = _prepare_color(out, color)

    x0, y0 = p1
    x1, y1 = p2

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        out = draw_point(out, (x0, y0), color, thickness)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return out


def draw_rectangle(
    img: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color,
    thickness: int = 1,
    filled: bool = False
) -> np.ndarray:
    """
    Draw rectangle on image.
    """
    validate_image(img)
    if not isinstance(thickness, int) or thickness <= 0:
        raise ValueError("thickness must be a positive integer.")

    out = img.copy().astype(np.float32)
    color = _prepare_color(out, color)

    x1, y1 = top_left
    x2, y2 = bottom_right

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(out.shape[1] - 1, x2)
    y2 = min(out.shape[0] - 1, y2)

    if filled:
        out[y1:y2 + 1, x1:x2 + 1] = color
        return out

    out = draw_line(out, (x1, y1), (x2, y1), color, thickness)
    out = draw_line(out, (x2, y1), (x2, y2), color, thickness)
    out = draw_line(out, (x2, y2), (x1, y2), color, thickness)
    out = draw_line(out, (x1, y2), (x1, y1), color, thickness)
    return out


def draw_polygon(img: np.ndarray, points: list[tuple[int, int]], color, thickness: int = 1) -> np.ndarray:
    """
    Draw polygon outline by connecting points.
    """
    validate_image(img)

    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("points must be a list with at least 3 points.")

    out = img.copy().astype(np.float32)
    for i in range(len(points)):
        out = draw_line(out, points[i], points[(i + 1) % len(points)], color, thickness)
    return out


def put_text(
    img: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 12,
    color="red"
) -> np.ndarray:
    """
    Place text on image using Matplotlib rendering then convert back to NumPy array.
    """
    validate_image(img)

    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(font_size, int) or font_size <= 0:
        raise ValueError("font_size must be a positive integer.")

    x, y = position

    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img, vmin=0, vmax=1)

    ax.text(x, y, text, fontsize=font_size, color=color)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    rendered = plt.imread(buf).astype(np.float32)
    if rendered.shape[2] == 4:
        rendered = rendered[..., :3]

    if img.ndim == 2:
        rendered = np.dot(rendered[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    return rendered
