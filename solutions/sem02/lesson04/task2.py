import numpy as np


def get_dominant_color_info(
    image: np.ndarray[np.uint8],
    threshold: int = 5,
) -> tuple[np.uint8, float]:
    # ваш код
    if threshold < 1:
        raise ValueError
    
    color_groups = {}
    
    h, w = image.shape
    total_pixels = h * w

    unique_values = np.unique(image)

    sorted_values = sorted(unique_values)
    
    groups = []
    current_group = [sorted_values[0]]
    
    for i in range(1, len(sorted_values)):
        if sorted_values[i] - sorted_values[i-1] < threshold:
            current_group.append(sorted_values[i])
        else:
            groups.append(current_group)
            current_group = [sorted_values[i]]
    groups.append(current_group)
    
    max_count = 0
    best_group = None
    
    for group in groups:
        count = sum(np.sum(image == val) for val in group)
        if count > max_count:
            max_count = count
            best_group = group
    
    dominant_color = int(round(np.mean(best_group)))
    percentage = (max_count / total_pixels) * 100
    
    return (dominant_color, percentage)
