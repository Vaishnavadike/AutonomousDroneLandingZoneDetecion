import torch



ORIGINAL_WIDTH = 800
ORIGINAL_HEIGHT = 600

MODEL_INPUT_SIZE = 256




NUM_CLASSES = 9

SAFE_CLASSES = [1, 4, 6]




CLASS_TO_TERRAIN = {
    1: "Pavement",
    3: "Grass",
    4: "Dirt"
}




BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 30




DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




DISTANCE_WEIGHT = 0.4
ROUGHNESS_WEIGHT = 0.2
SEMANTIC_WEIGHT = 0.4




BOX_WIDTH = 60
BOX_HEIGHT = 60
SEARCH_STEP = 10
MAX_SLOPE = 0.15