import cv2
import numpy as np


# ==========================================
# SAFE MASK FROM SEMANTIC LABEL IMAGE
# ==========================================

def create_safe_mask(semantic_image):

    safe_mask = np.zeros(
        semantic_image.shape[:2],
        dtype=np.uint8
    )

    # road color
    ROAD_COLOR = np.array([128, 64, 128])

    # grass color
    GRASS_COLOR = np.array([0, 102, 0])

    tolerance = 10


    # ==========================================
    # ROAD MASK
    # ==========================================

    road_mask = np.all(
        np.abs(semantic_image - ROAD_COLOR) < tolerance,
        axis=-1
    )


    # ==========================================
    # GRASS MASK
    # ==========================================

    grass_mask = np.all(
        np.abs(semantic_image - GRASS_COLOR) < tolerance,
        axis=-1
    )


    safe_mask[road_mask] = 255
    safe_mask[grass_mask] = 255


    # ==========================================
    # CLEAN MASK
    # ==========================================

    kernel = np.ones((7,7), np.uint8)

    safe_mask = cv2.morphologyEx(
        safe_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    safe_mask = cv2.morphologyEx(
        safe_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return safe_mask


# ==========================================
# ROUGHNESS COMPUTATION
# ==========================================

def compute_roughness(depth_map, box):

    x, y, w, h = box

    region = depth_map[
        y:y+h,
        x:x+w
    ]


    if region.size == 0:

        return 1.0


    sobel_x = cv2.Sobel(
        region,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        region,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )


    magnitude = np.sqrt(
        sobel_x**2 + sobel_y**2
    )


    roughness = np.mean(
        magnitude
    ) / 255.0


    return roughness