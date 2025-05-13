import os
import argparse
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from improved_cifar_training import ImprovedCNN

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a trained model from disk."""
    model = ImprovedCNN(class_count=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess a single image for inference."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4464), (.2023, .1994, .2010))
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_image(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> Tuple[str, float]:
    """Make prediction for a single image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return CLASS_NAMES[predicted.item()], confidence.item()

def predict_batch(model: nn.Module, image_tensors: List[torch.Tensor], device: torch.device) -> List[Tuple[str, float]]:
    """Make predictions for a batch of images."""
    batch_tensor = torch.cat(image_tensors, dim=0)
    with torch.no_grad():
        batch_tensor = batch_tensor.to(device)
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        return [(CLASS_NAMES[pred.item()], conf.item()) for pred, conf in zip(predicted, confidences)]

def interactive_mode(model_path: str, device: torch.device):
    """Interactive CLI mode for image inference."""
    model = load_model(model_path, device)
    print("\nCIFAR-10 Image Classifier")
    print("-------------------------")
    print("Enter 'q' to quit")
    
    while True:
        image_path = input("\nEnter image path: ").strip()
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found")
            continue
            
        try:
            image_tensor = preprocess_image(image_path)
            class_name, confidence = predict_image(model, image_tensor, device)
            print(f"\nPrediction: {class_name}")
            print(f"Confidence: {confidence:.2%}")
        except Exception as e:
            print(f"Error processing image: {str(e)}")

def batch_mode(model_path: str, image_dir: str, device: torch.device):
    """Process all images in a directory."""
    model = load_model(model_path, device)
    print(f"\nProcessing images in: {image_dir}")
    
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("No images found in directory")
        return
        
    print(f"Found {len(image_paths)} images")
    
    image_tensors = []
    valid_paths = []
    
    for path in image_paths:
        try:
            tensor = preprocess_image(path)
            image_tensors.append(tensor)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    if not image_tensors:
        print("No valid images to process")
        return
        
    predictions = predict_batch(model, image_tensors, device)
    
    print("\nResults:")
    print("--------")
    for path, (class_name, confidence) in zip(valid_paths, predictions):
        print(f"\nImage: {os.path.basename(path)}")
        print(f"Prediction: {class_name}")
        print(f"Confidence: {confidence:.2%}")

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classifier')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--batch', type=str, help='Directory containing images to process')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
        
    device = torch.device(args.device)
    
    if args.batch:
        batch_mode(args.model, args.batch, device)
    else:
        interactive_mode(args.model, device)

if __name__ == '__main__':
    main() 