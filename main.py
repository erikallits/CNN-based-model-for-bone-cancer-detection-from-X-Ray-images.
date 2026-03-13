# =====================
# 0. Import libraries
# =====================
import pandas as pd                  # For reading CSV files
import os                            # For handling file paths
from PIL import Image                # For opening images
from torch.utils.data import Dataset, DataLoader  # For PyTorch dataset management
import torchvision.transforms as transforms       # For image transformations
import torch                          # PyTorch main library
import torch.nn as nn                 # For building neural networks
import torch.optim as optim           # Optimizers for training

# =====================
# 1. Image Transformations
# =====================
# Transformations for training dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize all images to 224x224
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
    transforms.ToTensor(),            # Convert PIL image to PyTorch tensor (C,H,W) and scale to [0,1]
])

# Transformations for validation dataset
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Transformations for test dataset
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================
# 2. Custom Dataset Class
# =====================
class BoneCancerDataset(Dataset):
    """
    PyTorch Dataset for Bone Cancer X-Ray Images.
    Uses a CSV file for labels and a folder for images.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        # Load CSV file
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Remove any whitespace from column headers
        self.data.columns = self.data.columns.str.strip()

        # Create a label column: cancer=1, normal=0
        if 'cancer' in self.data.columns:
            self.data['label'] = self.data['cancer'].astype(int)
        elif 'normal' in self.data.columns:
            self.data['label'] = 1 - self.data['normal'].astype(int)  # normal=1 -> label=0
        else:
            raise ValueError("CSV must contain either 'cancer' or 'normal' column")

    def __len__(self):
        # Return total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single sample: (image, label)
        """
        # Get image path
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])
        # Open image and convert to RGB
        image = Image.open(img_name).convert("RGB")
        # Get label
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        return image, label

# =====================
# 3. Dataset paths
# =====================
train_csv = 'dataset/train/_classes.csv'
valid_csv = 'dataset/valid/_classes.csv'
test_csv  = 'dataset/test/_classes.csv'

train_img_dir = 'dataset/train'
valid_img_dir = 'dataset/valid'
test_img_dir  = 'dataset/test'

# =====================
# 4. Create Dataset objects and DataLoaders
# =====================
train_dataset = BoneCancerDataset(train_csv, train_img_dir, train_transform)
valid_dataset = BoneCancerDataset(valid_csv, valid_img_dir, val_transform)
test_dataset  = BoneCancerDataset(test_csv, test_img_dir, test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Quick check of a batch
for imgs, labels in train_loader:
    print(imgs.shape, labels.shape)
    break

# =====================
# 5. Define CNN Model
# =====================
class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for bone cancer detection.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Feature extractor: convolutional layers + pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        # Classifier: fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*28*28, 128),  # Adjusted for input image 224x224
            nn.ReLU(),
            nn.Linear(128, 2)          # 2 classes: cancer / normal
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =====================
# 6. Set device, loss function, optimizer
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()   # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 7. Training loop
# =====================
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()        # Clear gradients
        outputs = model(imgs)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()              # Backpropagation
        optimizer.step()             # Update weights

        # Compute running statistics
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    # Validation
    model.eval()
    val_corrects = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels).item()
            val_total += labels.size(0)
    val_acc = val_corrects / val_total

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
          f"Val Acc: {val_acc:.4f}")

# =====================
# 8. Testing / Evaluation
# =====================
model.eval()
test_corrects = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels).item()
        test_total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_corrects / test_total
print(f"Test Accuracy: {test_acc:.4f}")

# =====================
# 9. Preview predictions for a few images
# =====================
for i in range(10):
    print(f"Image {i+1}: True Label = {all_labels[i]}, Predicted = {all_preds[i]}")