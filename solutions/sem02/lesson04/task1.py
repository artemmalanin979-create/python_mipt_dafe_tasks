import numpy as np


def pad_image(image: np.ndarray, pad_size: int) -> np.ndarray:
    # ваш код
    if pad_size < 1:
        raise ValueError

    if image.ndim == 2:
        h, w = image.shape
        padded_image = np.zeros((h + 2 * pad_size, w + 2 * pad_size), dtype=image.dtype)
        padded_image[pad_size : pad_size + h, pad_size : pad_size + w] = image
    elif image.ndim == 3:
        h, w, c = image.shape
        padded_image = np.zeros((h + 2 * pad_size, w + 2 * pad_size, c), dtype=image.dtype)
        padded_image[pad_size : pad_size + h, pad_size : pad_size + w, :] = image
    else:
        raise ValueError

    return padded_image


def blur_image(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    # ваш код
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError

    if kernel_size == 1:
        return image.copy()

    pad_size = kernel_size // 2
    h, w = image.shape[:2]
    padded = pad_image(image, pad_size)
    window_area = kernel_size * kernel_size

    if image.ndim == 2:
        integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.uint64)
        integral[1:, 1:] = np.cumsum(
            np.cumsum(padded, axis=0, dtype=np.uint64), axis=1, dtype=np.uint64
        )

        A = integral[:h, :w]
        B = integral[:h, kernel_size : kernel_size + w]
        C = integral[kernel_size : kernel_size + h, :w]
        D = integral[kernel_size : kernel_size + h, kernel_size : kernel_size + w]

        window_sums = D - C - B + A
        return (window_sums / window_area).astype(np.uint8)

    integral = np.zeros(
        (padded.shape[0] + 1, padded.shape[1] + 1, padded.shape[2]), dtype=np.uint64
    )
    integral[1:, 1:, :] = np.cumsum(
        np.cumsum(padded, axis=0, dtype=np.uint64), axis=1, dtype=np.uint64
    )

    A = integral[:h, :w, :]
    B = integral[:h, kernel_size : kernel_size + w, :]
    C = integral[kernel_size : kernel_size + h, :w, :]
    D = integral[kernel_size : kernel_size + h, kernel_size : kernel_size + w, :]

    window_sums = D - C - B + A
    return (window_sums / window_area).astype(np.uint8)


if __name__ == "__main__":
    import os
    from pathlib import Path

    from utils.utils import compare_images, get_image

    current_directory = Path(__file__).resolve().parent
    image = get_image(os.path.join(current_directory, "images", "circle.jpg"))
    image_blured = blur_image(image, kernel_size=21)

    compare_images(image, image_blured)
