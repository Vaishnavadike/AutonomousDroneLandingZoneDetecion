import json

from collections import Counter

from config import CLASS_TO_TERRAIN


MISSION_MAPPING = {
    "fragile": "Fragile",
    "valuable": "Valuable",
    "biohazard": "Biohazard",
    "heavy": "Heavy"
}


def load_active_nodes(config_path):

    with open(config_path, 'r') as f:
        data = json.load(f)

    payload = data['payload']

    active = []

    for key, value in payload.items():

        if value:
            active.append(MISSION_MAPPING[key])

    return active

def dominant_terrain(segmentation_map, box):

    x, y, w, h = box

    region = segmentation_map[y:y+h, x:x+w]

    values = region.flatten()

    values = [v for v in values if v in CLASS_TO_TERRAIN]

    dominant = Counter(values).most_common(1)[0][0]

    return CLASS_TO_TERRAIN[dominant]