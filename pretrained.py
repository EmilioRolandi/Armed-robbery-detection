import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from collections import Counter

# Parameters
DATA_DIR = "output_folder"
BATCH_SIZE = 16
EPOCHS = 20
LR_HEAD = 0.001      # Learning rate for the head
LR_FINE = 1e-4       # Learning rate for fine-tuning
NUM_CLASSES = 2

# Count class distribution
files = [f for f in os.listdir(DATA_DIR) if f.endswith((".png",".jpg",".jpeg"))]
labels = [1 if f.lower().startswith("y") else 0 for f in files]
counts = Counter(labels)
total = len(files)
print("Class distribution:")
print(f"True (y): {counts[1]} images")
print(f"False (n): {counts[0]} images")
print(f"Percentage True: {counts[1]/total*100:.2f}%")
print(f"Percentage False: {counts[0]/total*100:.2f}%")

# Custom Dataset
class TheftDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith((".png",".jpg",".jpeg"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        label = 1 if file_name.lower().startswith("y") else 0
        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = TheftDataset(DATA_DIR, transform=transform)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# Load pretrained ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Replace last layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.fc.requires_grad = True  # Only this layer trains first
model = model.to(device)

# Loss and optimizer (only for the head first)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD)

unfreeze_schedule = {
    5: 'layer3',   # unfreeze layer3 at epoch 5
    10: 'layer2',  # unfreeze layer2 at epoch 10
}

# Training loop
for epoch in range(EPOCHS):
    
    # Check if we need to unfreeze a layer this epoch
    if epoch in unfreeze_schedule:
        layer_to_unfreeze = getattr(model, unfreeze_schedule[epoch])
        for param in layer_to_unfreeze.parameters():
            param.requires_grad = True
        print(f"Unfroze {unfreeze_schedule[epoch]} at epoch {epoch}")
        
        # Optionally, re-create the optimizer with the new trainable parameters
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINE)
    
    # Training
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
    
    # Testing
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.2%}")

# Save the model state_dict
torch.save(model.state_dict(), "resnet18_theft.pth")

