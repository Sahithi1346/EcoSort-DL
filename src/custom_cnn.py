"""
EcoSort AI - Custom CNN Model Architecture
Built from scratch with Convolutional layers, Batch Normalization, ReLU activations,
Max Pooling, Adaptive Average Pooling, Dropout regularization, and Fully Connected classifier.
"""

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    Custom 4-stage Convolutional Neural Network designed for 6-class waste classification.
    Input resolution: (3, 224, 224)
    Output: 6 class logits
    """

    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.3):
        super(CustomCNN, self).__init__()

        # Conv Block 1: (3, 224, 224) -> (32, 112, 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1)
        )

        # Conv Block 2: (32, 112, 112) -> (64, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15)
        )

        # Conv Block 3: (64, 56, 56) -> (128, 28, 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )

        # Conv Block 4: (128, 28, 28) -> (256, 2, 2) via AdaptiveAvgPool
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Dropout(p=0.25)
        )

        # Classifier: 256 * 2 * 2 = 1024 -> 256 -> 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits


def build_custom_cnn(num_classes: int = 6, dropout_rate: float = 0.3) -> CustomCNN:
    """Helper factory function to create a CustomCNN instance."""
    return CustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)


if __name__ == "__main__":
    model = build_custom_cnn(num_classes=6)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print("CustomCNN initialized successfully!")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
