import os
import glob
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import GrazDataset
from model import UNetSmall
from config import *


TRAIN_IMAGE_DIR = "/content/drive/MyDrive/CSF407_2026_ID/semantic_drone_dataset_semantics_v1.1/semantic_drone_dataset/training_set/images"

TRAIN_MASK_DIR = "/content/drive/MyDrive/CSF407_2026_ID/semantic_drone_dataset_semantics_v1.1/semantic_drone_dataset/training_set/gt/semantic/label_images"


train_images = sorted(
    glob.glob(
        os.path.join(TRAIN_IMAGE_DIR, "*.jpg")
    )
)

train_masks = sorted(
    glob.glob(
        os.path.join(TRAIN_MASK_DIR, "*.png")
    )
)


print(f"Images: {len(train_images)}")
print(f"Masks: {len(train_masks)}")


train_dataset = GrazDataset(
    train_images,
    train_masks
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


model = UNetSmall(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_loss = float('inf')


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, masks in train_loader:

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "/content/drive/MyDrive/CSF407_2026_ID/src/best_model.pth"
        )

        print("Best model saved")


print("Training Completed")