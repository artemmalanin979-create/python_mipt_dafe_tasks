import numpy as np


def get_dominant_color_info(
    image: np.ndarray[np.uint8],
    threshold: int = 5,
) -> tuple[np.uint8, float]:
    if threshold < 1:
        raise ValueError("threshold must be positive")

    total_pixels = image.size
    histogram = np.bincount(image.flatten(), minlength=256)
    prefix_sums = np.zeros(257, dtype=np.int64)
    prefix_sums[1:] = np.cumsum(histogram)

    delta = threshold - 1
    unique_values = np.unique(image)

    max_count = -1
    dominant_color = np.uint8(0)

    for value in unique_values:
        color = int(value)
        left = max(0, color - delta)
        right = min(255, color + delta)
        count = prefix_sums[right + 1] - prefix_sums[left]

        if count > max_count:
            max_count = count
            dominant_color = np.uint8(color)

    percentage = (max_count / total_pixels) * 100

    return dominant_color, percentage
