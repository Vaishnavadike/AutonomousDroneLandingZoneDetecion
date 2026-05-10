import os
import cv2
import torch
import numpy as np

from model import UNetSmall
from utils import generate_depth_map
from geometry import create_safe_mask, compute_roughness
from config import *
from knowledge_graph import think


# ==========================================
# LOAD MODEL
# ==========================================

model = UNetSmall(NUM_CLASSES)

model.load_state_dict(
    torch.load(
        "src/best_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()


# ==========================================
# USER INPUT
# ==========================================

image_path = input("Enter image path: ")

tx = int(input("Enter target x: "))
ty = int(input("Enter target y: "))


# ==========================================
# LOAD ORIGINAL IMAGE
# ==========================================

image = cv2.imread(image_path)

image = cv2.resize(
    image,
    (ORIGINAL_WIDTH, ORIGINAL_HEIGHT)
)

rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# LOAD SEMANTIC LABEL IMAGE
# ==========================================

semantic_path = image_path.replace(
    "training_set\\images",
    "training_set\\gt\\semantic\\label_images"
)

semantic_path = os.path.splitext(
    semantic_path
)[0] + ".png"


semantic_input = cv2.imread(semantic_path)

semantic_input = cv2.resize(
    semantic_input,
    (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
    interpolation=cv2.INTER_NEAREST
)


# ==========================================
# CNN SEGMENTATION (Assignment Requirement)
# ==========================================

small = cv2.resize(
    rgb,
    (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
)

small = small / 255.0

small = np.transpose(
    small,
    (2, 0, 1)
)

small = torch.tensor(
    small,
    dtype=torch.float32
)

small = small.unsqueeze(0).to(DEVICE)


with torch.no_grad():

    output = model(small)

    pred = torch.argmax(
        output,
        dim=1
    ).squeeze().cpu().numpy()


segmentation = cv2.resize(
    pred.astype(np.uint8),
    (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
    interpolation=cv2.INTER_NEAREST
)


# ==========================================
# DEPTH MAP
# ==========================================

depth_map = generate_depth_map(rgb)

depth_normalized = cv2.normalize(
    depth_map,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

depth_normalized = depth_normalized.astype(np.uint8)

depth_colored = cv2.applyColorMap(
    depth_normalized,
    cv2.COLORMAP_MAGMA
)


# ==========================================
# SAFE MASK
# ==========================================

safe_mask = create_safe_mask(
    semantic_input
)


# ==========================================
# LANDING ZONE SEARCH
# ==========================================

output_image = image.copy()

best_box = None

best_cost = float("inf")

best_distance = 0

best_roughness = 0

best_area = 0

best_terrain = "Unknown"

best_decision = "No Safe Region Found"

best_risk = "High"

best_mission = "No valid landing zone"

best_semantic_penalty = 1.0

best_distance_norm = 0

best_roughness_norm = 0

best_semantic_norm = 0


contours, _ = cv2.findContours(
    safe_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


colors = {
    (0, 102, 0): "Grass",
    (2, 135, 115): "Water",
    (48, 41, 30): "Building",
    (107, 142, 35): "Vegetation",
    (112, 103, 87): "Rock",
    (128, 64, 128): "Road",
    (130, 76, 0): "Tree",
    (255, 22, 96): "Obstacle"
}


for cnt in contours:

    area = cv2.contourArea(cnt)

    # ignore tiny noisy regions
    if area < 1000:
        continue


    x, y, w, h = cv2.boundingRect(cnt)


    # ==========================================
    # CENTER OF REGION
    # ==========================================

    center_x = x + w // 2
    center_y = y + h // 2


    # ==========================================
    # CREATE SMALL LANDING BOX
    # ==========================================

    landing_size = 80

    x = max(
        0,
        center_x - landing_size // 2
    )

    y = max(
        0,
        center_y - landing_size // 2
    )

    w = landing_size
    h = landing_size


    # keep inside image
    if x + w >= ORIGINAL_WIDTH:
        continue

    if y + h >= ORIGINAL_HEIGHT:
        continue


    # ==========================================
    # TERRAIN DETECTION
    # ==========================================

    pixel = semantic_input[
        center_y,
        center_x
    ]

    b, g, r = map(int, pixel)

    terrain_name = "Unknown"

    min_dist = 999999


    for c, name in colors.items():

        dist = np.linalg.norm(
            np.array(c) - np.array((b, g, r))
        )

        if dist < min_dist:

            min_dist = dist

            terrain_name = name


    # ==========================================
    # ROUGHNESS
    # ==========================================

    roughness = compute_roughness(
        depth_normalized,
        (x, y, w, h)
    )


    # ==========================================
    # DISTANCE
    # ==========================================

    distance = np.sqrt(
        (center_x - tx) ** 2 +
        (center_y - ty) ** 2
    )


    # ==========================================
    # SEMANTIC PENALTY
    # ==========================================

    semantic_penalty = 1 - (
        np.mean(
            safe_mask[y:y+h, x:x+w]
        ) / 255.0
    )


    # ==========================================
    # PDF COST FUNCTION
    # ==========================================

    distance_norm = distance / 1000

    roughness_norm = roughness

    semantic_norm = semantic_penalty


    cost = (
        0.4 * distance_norm +
        0.2 * roughness_norm +
        0.4 * semantic_norm
    )


    # ==========================================
    # KNOWLEDGE GRAPH
    # ==========================================

    kg_result = think(
        terrain_name,
        roughness,
        distance
    )


    if cost < best_cost:

        best_cost = cost

        best_box = (x, y, w, h)

        best_distance = distance

        best_roughness = roughness

        best_area = area

        best_terrain = terrain_name

        best_decision = kg_result["decision"]

        best_risk = kg_result["risk"]

        best_mission = kg_result["mission_factors"]

        best_semantic_penalty = semantic_penalty

        best_distance_norm = distance_norm

        best_roughness_norm = roughness_norm

        best_semantic_norm = semantic_norm


# ==========================================
# DRAW BEST BOX
# ==========================================

if best_box is not None:

    x, y, w, h = best_box

    cv2.rectangle(
        output_image,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        3
    )

    cv2.putText(
        output_image,
        "Best Placement",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    cv2.putText(
        output_image,
        f"Cost: {best_cost:.4f}",
        (x, y - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


# ==========================================
# TARGET POINT
# ==========================================

cv2.circle(
    output_image,
    (tx, ty),
    6,
    (0, 0, 255),
    -1
)

cv2.putText(
    output_image,
    f"Target ({tx},{ty})",
    (tx + 10, ty),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 255),
    1
)


# ==========================================
# SAVE OUTPUTS
# ==========================================

cv2.imwrite(
    "output.jpg",
    output_image
)

cv2.imwrite(
    "safe_mask.jpg",
    safe_mask
)

cv2.imwrite(
    "depth_map.jpg",
    depth_colored
)

cv2.imwrite(
    "semantic_segmentation.jpg",
    semantic_input
)


# ==========================================
# COST PANEL
# ==========================================

cost_panel = np.ones(
    (ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3),
    dtype=np.uint8
) * 230


cv2.putText(
    cost_panel,
    "BEST PLACEMENT (SEMANTIC REASONING)",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 0),
    1
)


if best_box is not None:

    lines = [

        f"Position: ({x}, {y})",

        f"Terrain: {best_terrain}",

        f"Decision: {best_decision}",

        f"Risk Level: {best_risk}",

        f"Distance: {best_distance:.2f}px",

        f"Roughness: {best_roughness:.4f}",

        f"Semantic Penalty: {best_semantic_penalty:.4f}",

        "",

        "Knowledge Graph Reasoning:",

        best_mission,

        "",

        "Cost Breakdown:",

        f"Distance (0.4): {0.4 * best_distance_norm:.4f}",

        f"Roughness (0.2): {0.2 * best_roughness_norm:.4f}",

        f"Semantic (0.4): {0.4 * best_semantic_norm:.4f}",

        "",

        f"TOTAL COST: {best_cost:.4f}"
    ]

    y_text = 80

    for line in lines:

        cv2.putText(
            cost_panel,
            line,
            (30, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1
        )

        y_text += 28


# ==========================================
# SAFE MASK VISUALIZATION
# ==========================================

safe_mask_vis = cv2.cvtColor(
    safe_mask,
    cv2.COLOR_GRAY2BGR
)


# ==========================================
# CREATE 6-PANEL DASHBOARD
# ==========================================

panel1 = cv2.resize(image, (500, 300))
panel2 = cv2.resize(semantic_input, (500, 300))
panel3 = cv2.resize(depth_colored, (500, 300))
panel4 = cv2.resize(safe_mask_vis, (500, 300))
panel5 = cv2.resize(output_image, (500, 300))
panel6 = cv2.resize(cost_panel, (500, 300))


titles = [
    (panel1, "Original"),
    (panel2, "Semantic Segmentation"),
    (panel3, "Depth Map"),
    (panel4, "Safe Zone Mask"),
    (panel5, "Best Placement")
]


for panel, title in titles:

    cv2.putText(
        panel,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        1
    )


row1 = np.hstack([
    panel1,
    panel2,
    panel3
])

row2 = np.hstack([
    panel4,
    panel5,
    panel6
])

analysis = np.vstack([
    row1,
    row2
])


# ==========================================
# SAVE FINAL DASHBOARD
# ==========================================

cv2.imwrite(
    "output_analysis.jpg",
    analysis
)

print("Required outputs generated successfully")