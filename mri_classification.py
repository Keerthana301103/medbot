import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels
class_names = ["glioma", "meningioma", "pituitary", "notumor"]

# Load EfficientNet model
model = EfficientNet.from_pretrained("efficientnet-b0")
model._fc = nn.Linear(model._fc.in_features, len(class_names))  # 4-class classification
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()
def predict_mri(image_path):
    """Predicts MRI class from an uploaded image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Ensure gradients can be computed
    image_tensor.requires_grad = True  

    # Run model with gradient tracking
    output = model(image_tensor)  

    # Get predicted class
    _, predicted_class = torch.max(output, 1)
    prediction = class_names[predicted_class.item()]

    return prediction, output 