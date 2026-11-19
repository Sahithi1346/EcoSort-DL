"""
EcoSort AI - ResNet-18 Transfer Learning Model
Loads pretrained ResNet-18 weights, modifies the final classification head for 6 classes,
and configures fine-tuning layers.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EcoSortResNet18(nn.Module):
    """
    ResNet-18 adapted for 6-class EcoSort waste classification.
    Leverages ImageNet pretrained representations with a custom classification head.
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False):
        super(EcoSortResNet18, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Optional backbone freezing for faster CPU fine-tuning
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                # Unfreeze layer4 and final fc
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        in_features = self.backbone.fc.in_features
        # Custom head with Dropout regularization
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_resnet18(num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False) -> EcoSortResNet18:
    """Helper factory function to create an EcoSortResNet18 instance."""
    return EcoSortResNet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    model = build_resnet18(num_classes=6, pretrained=True, freeze_backbone=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("ResNet-18 initialized successfully!")
    print(f"Output shape: {out.shape}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable / Total params: {trainable_params:,} / {total_params:,}")
