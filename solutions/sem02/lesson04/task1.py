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
    
    pad_size = kernel_size // 2
    padded_image = pad_image(image, pad_size)

    if image.ndim == 2:
        h, w = image.shape
        blur_image = np.zeros_like(image, dtype=np.float64)

        for i in range(h):
            for j in range(w):
                blur_image[i, j] = np.mean(padded_image[i : i + kernel_size, j : j + kernel_size])
    
    elif image.ndim == 3:
        h, w, c = image.shape
        blur_image = np.zeros_like(image, dtype=np.float64)

        for i in range(h): 
            for j in range(w):
                    blur_image[i, j, :] = np.mean(padded_image[i : i + kernel_size, j : j + kernel_size, :], axis=(0, 1))
        
    image = blur_image.astype(np.uint8)
    return image


if __name__ == "__main__":
    import os
    from pathlib import Path

    from utils.utils import compare_images, get_image

    current_directory = Path(__file__).resolve().parent
    image = get_image(os.path.join(current_directory, "images", "circle.jpg"))
    image_blured = blur_image(image, kernel_size=21)

    compare_images(image, image_blured)
