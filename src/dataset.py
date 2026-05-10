import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from config import MODEL_INPUT_SIZE


CLASS_COLORS = {
    (0, 0, 0): 0,
    (0, 102, 0): 1,
    (2, 135, 115): 2,
    (48, 41, 30): 3,
    (107, 142, 35): 4,
    (112, 103, 87): 5,
    (128, 64, 128): 6,
    (130, 76, 0): 7,
    (255, 22, 96): 8
}


class GrazDataset(Dataset):

    def __init__(self, image_paths, mask_paths):

        self.image_paths = image_paths
        self.mask_paths = mask_paths

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.image_paths)

    def rgb_to_class(self, mask):

        label = np.zeros(
            (mask.shape[0], mask.shape[1]),
            dtype=np.uint8
        )

        for rgb, class_id in CLASS_COLORS.items():

            matches = np.all(
                mask == rgb,
                axis=-1
            )

            label[matches] = class_id

        return label

    def __getitem__(self, idx):

        image = cv2.imread(
            self.image_paths[idx]
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            self.mask_paths[idx]
        )

        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        )

        mask = cv2.resize(
            mask,
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        mask = self.rgb_to_class(mask)

        image = self.transform(image)

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image, mask