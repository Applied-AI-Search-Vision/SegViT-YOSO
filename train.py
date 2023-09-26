from Attention import *
from Block import *
from DropPath import *
from MLP import *
from PatchEmbed import *
from YOSOHead import YOSOHead

import torch
import torch.nn as nn
import torch.optim as optim


# Detectron2 imports
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

setup_logger()


class SegViT_YOSO(nn.Module):
    def __init__(self, image_size, patch_size, num_classes):
        super(SegViT_YOSO, self).__init__()
        self.patch_embed = PatchEmbed(image_size=image_size, patch_size=patch_size)
        self.blocks = nn.Sequential(*[Block() for _ in range(12)])  # Adjust the number of blocks if needed
        self.head = YOSOHead()  # This is the YOSO part

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        yoso_out = self.head(x)  # YOSO head processes the ViT blocks' output
        return yoso_out


# Define the combined loss function for SegViT
class SegViTLoss(nn.Module):
    def __init__(self):
        super(SegViTLoss, self).__init__()

    def forward(self, outputs, targets):
        masks_pred, class_scores = outputs
        masks_true, class_labels = targets
        segmentation_loss = nn.functional.binary_cross_entropy_with_logits(masks_pred, masks_true, reduction='mean')
        classification_loss = nn.functional.cross_entropy(class_scores, class_labels, reduction='mean')
        combined_loss = segmentation_loss + classification_loss
        return combined_loss


# Register COCO dataset with Detectron2
train_data_dir = "C:/Users/stellan.lange/OneDrive - NIOUSA/Desktop/RL_tasks/SegViT-YOSO/COCODataset/train2017"
train_json_dir = "C:/Users/stellan.lange/OneDrive - NIOUSA/Desktop/RL_tasks/SegViT-YOSO/COCODataset/annotations/instances_train2017.json"
register_coco_instances("coco_train", {}, train_json_dir, train_data_dir)

# Detectron2's configuration and default setup
cfg = get_cfg()
cfg.DATASETS.TRAIN = ("coco_train",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 32
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

# Model, loss, and optimizer instantiation
model = SegViT_YOSO(image_size=224, patch_size=16, num_classes=80)
model = model.to('cuda')
criterion = SegViTLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    data_loader = build_detection_train_loader(cfg)
    for i, batch in enumerate(data_loader):
        images = batch['image'].to('cuda')
        targets = batch['instances'].to('cuda')
        outputs = model(images)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

print('Finished Training')
