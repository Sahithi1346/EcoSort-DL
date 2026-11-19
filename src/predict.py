"""
EcoSort AI - Inference & Prediction Engine
Loads model checkpoints, applies standard image preprocessing, computes softmax probabilities,
and connects predictions to the recommendation engine.
"""

import os
import sys
import io

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Union
from torchvision import transforms

from src.custom_cnn import build_custom_cnn
from src.resnet_model import build_resnet18
from src.densenet_model import build_densenet121
from src.recommendations import get_recommendation
from src.dataset_loader import CLASSES, IMAGENET_MEAN, IMAGENET_STD

MODELS_DIR = "models"

# Standard inference transform
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Global model cache to avoid reloading weights repeatedly in Streamlit
_LOADED_MODELS: Dict[str, torch.nn.Module] = {}


def get_cached_model(model_name: str, device: torch.device = None) -> torch.nn.Module:
    """Loads and caches the model checkpoint in memory for fast repeated inference."""
    global _LOADED_MODELS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_name = model_name.lower().strip()
    cache_key = f"{clean_name}_{device.type}"

    if cache_key in _LOADED_MODELS:
        return _LOADED_MODELS[cache_key]

    ckpt_path = os.path.join(MODELS_DIR, f"{clean_name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    # Build architecture
    if clean_name in ["custom_cnn", "cnn"]:
        model = build_custom_cnn(num_classes=len(CLASSES))
    elif clean_name in ["resnet18", "resnet"]:
        model = build_resnet18(num_classes=len(CLASSES), pretrained=False)
    elif clean_name in ["densenet121", "densenet"]:
        model = build_densenet121(num_classes=len(CLASSES), pretrained=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _LOADED_MODELS[cache_key] = model
    return model


def predict_waste_image(
    image_input: Union[str, bytes, Image.Image],
    model_name: str = "resnet18",
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Takes an image file path, raw bytes, or PIL Image,
    runs inference using the specified model, and returns
    predicted category, confidence, all class probabilities,
    dustbin recommendation, disposal instructions, and environmental tip.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load PIL image
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # Preprocess tensor
    input_tensor = INFERENCE_TRANSFORM(pil_img).unsqueeze(0).to(device)

    # Inference
    model = get_cached_model(model_name, device=device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probabilities))
    pred_class = CLASSES[pred_idx]
    confidence = float(probabilities[pred_idx]) * 100.0

    class_probabilities = {
        cls_name: round(float(prob) * 100.0, 2)
        for cls_name, prob in zip(CLASSES, probabilities)
    }

    # Retrieve recommendation and disposal guidance
    rec_info = get_recommendation(pred_class)

    return {
        "predicted_class": pred_class,
        "predicted_title": rec_info["category_title"],
        "confidence": round(confidence, 2),
        "class_probabilities": class_probabilities,
        "model_used": model_name,
        "dustbin": rec_info["dustbin"],
        "dustbin_color": rec_info["dustbin_color"],
        "bin_type": rec_info["bin_type"],
        "bin_icon": rec_info["bin_icon"],
        "verification_required": rec_info["verification_required"],
        "instructions": rec_info["instructions"],
        "detailed_steps": rec_info.get("detailed_steps", rec_info["instructions"]),
        "environmental_tip": rec_info["environmental_tip"],
        "recyclability": rec_info["recyclability"],
        "decomposition_time": rec_info.get("decomposition_time", "N/A"),
        "material_profile": rec_info.get("material_profile", ""),
        "accepted_examples": rec_info.get("accepted_examples", []),
        "rejected_examples": rec_info.get("rejected_examples", []),
        "recycling_process": rec_info.get("recycling_process", ""),
        "environmental_savings": rec_info.get("environmental_savings", {}),
    }


if __name__ == "__main__":
    print("Inference module ready. CLASSES:", CLASSES)
