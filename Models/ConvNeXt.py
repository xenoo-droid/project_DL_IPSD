"""Model 3: ConvNeXt-Tiny pretrained ImageNet, fine-tune semua layer buat 6 kelas."""

import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # classifier aslinya [LayerNorm2d, Flatten, Linear(768, 1000)],
        # layer terakhir diganti jadi 768 -> num_classes
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # output logits, softmax udah diurus CrossEntropyLoss
        return self.model(x)
