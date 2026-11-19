"""
EcoSort AI - DenseNet-121 Transfer Learning Model
Loads pretrained DenseNet-121 weights, adapts the final classifier for 6 classes,
and provides fine-tuning configurations.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EcoSortDenseNet121(nn.Module):
    """
    DenseNet-121 adapted for 6-class EcoSort waste classification.
    Exploits dense connectivity and feature reuse with a tailored classification head.
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False):
        super(EcoSortDenseNet121, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Optional backbone freezing for efficient CPU fine-tuning
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                # Keep denseblock4, norm5, and classifier trainable
                if "denseblock4" not in name and "norm5" not in name and "classifier" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        in_features = self.backbone.classifier.in_features
        # Replace classifier with Dropout + Linear layer
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_densenet121(num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False) -> EcoSortDenseNet121:
    """Helper factory function to create an EcoSortDenseNet121 instance."""
    return EcoSortDenseNet121(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    model = build_densenet121(num_classes=6, pretrained=True, freeze_backbone=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("DenseNet-121 initialized successfully!")
    print(f"Output shape: {out.shape}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable / Total params: {trainable_params:,} / {total_params:,}")
