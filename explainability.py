import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchcam.methods import GradCAMpp
from efficientnet_pytorch import EfficientNet
from PIL import Image

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
model = EfficientNet.from_pretrained("efficientnet-b0")
model._fc = torch.nn.Linear(model._fc.in_features, 4)  # 4-class classification
model.load_state_dict(torch.load(r"C:\Users\s.anumandla\Desktop\medBot\model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Initialize Grad-CAM++ on the last convolutional layer
target_layer = model._conv_head
cam_extractor = GradCAMpp(model, target_layer)

def generate_gradcam_explanation(image_path):
    """Generates Grad-CAM++ heatmap for the given MRI image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Get model output
    output = model(image_tensor)

    # Get predicted class index
    class_idx = torch.argmax(output, dim=1).item()

    # Generate Grad-CAM++ heatmap
    activation_map = cam_extractor(class_idx, output)

    # Convert activation map to NumPy array
    activation_map = activation_map[0].squeeze(0).cpu().numpy()  # ✅ Fix shape

    # Resize activation map to match original image size
    activation_map = cv2.resize(activation_map, (224, 224))  # ✅ Resize to input shape

    # Normalize activation map
    activation_map = np.interp(activation_map, (activation_map.min(), activation_map.max()), (0, 255))
    activation_map = np.uint8(activation_map)

    # Plot Grad-CAM overlay
    fig, ax = plt.subplots()
    ax.imshow(image)  # Original image
    ax.imshow(activation_map, cmap="jet", alpha=0.5)  # Overlay heatmap
    ax.axis("off")

    return fig
