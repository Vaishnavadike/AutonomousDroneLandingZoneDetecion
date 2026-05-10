import torch
import cv2
import numpy as np


# ==========================================
# LOAD MIDAS MODEL
# ==========================================

midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

midas.eval()


# ==========================================
# LOAD TRANSFORMS
# ==========================================

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform


# ==========================================
# DEPTH MAP GENERATION
# ==========================================


def generate_depth_map(image):

    input_batch = transform(image)

    with torch.no_grad():

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = prediction.cpu().numpy()

    depth = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth = depth.astype(np.uint8)

    return depth