import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import os
import glob
from PIL import Image

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match EfficientNet input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Define class names
class_names = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]
num_classes = len(class_names)

# Custom dataset class
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Dataset paths
train_dir = r"C:\Users\s.anumandla\Desktop\medBot\archive (1)\Training"
test_dir = r"C:\Users\s.anumandla\Desktop\medBot\archive (1)\Testing"

# Main training function
def train_and_save_model():
    # Load datasets
    train_dataset = MRIDataset(train_dir, transform=transform)
    test_dataset = MRIDataset(test_dir, transform=transform)

    # DataLoaders (set num_workers=0 for Windows)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Load EfficientNet model
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(model._fc.in_features, num_classes)  # Modify last layer for 4 classes
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

    # Save model
    model_path = os.path.join(r"C:\Users\s.anumandla\Desktop\medBot", "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    # Evaluation on Test Set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

# Ensure safe multiprocessing on Windows
if __name__ == '__main__':
    train_and_save_model()
