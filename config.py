import torch

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
MRI_MODEL_PATH = "brain_tumor_model.pth"
OLLAMA_MODEL = "mistral"

# Upload directories
UPLOAD_FOLDER = "uploads/"
