import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import numpy as np

# --- Check and set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Define MNIST transforms ---
mnist_transform = transforms.ToTensor()

# --- Only test set needed here initially (used for example predictions later) ---
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# --- Define the CNN model ---
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNNNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

MODEL_PATH = "mnist_cnn.pth"

# --- Load trained model or train a new one ---
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model weights from {MODEL_PATH}")
else:
    # Only load training data if model doesn't exist
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# --- Show predictions on sample test images ---
model.eval()
examples = enumerate(test_loader)
_, (example_data, example_targets) = next(examples)
example_data = example_data.to(device)

with torch.no_grad():
    output = model(example_data)

fig = plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    img = example_data[i].cpu().numpy().squeeze()
    plt.imshow(img, cmap='gray', interpolation='none')
    pred_label = output[i].argmax().item()
    true_label = example_targets[i].item()
    plt.title(f"Pred: {pred_label}, True: {true_label}")
    plt.axis('off')
plt.show()

# --- Predict your own image ---
def center_image(img):
    np_img = np.array(img)
    np_img[np_img < 20] = 0
    coords = np.argwhere(np_img > 0)
    if coords.size == 0:
        return img.resize((28, 28))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img.crop((x0, y0, x1, y1))
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
    centered_img = Image.new('L', (28, 28), color=0)
    x_offset = (28 - cropped.width) // 2
    y_offset = (28 - cropped.height) // 2
    centered_img.paste(cropped, (x_offset, y_offset))
    return centered_img

img_path = "HW-Nums\\digit1.png"  #put image path here
img = Image.open(img_path).convert('L')
img = ImageOps.invert(img)
img = center_image(img)

plt.imshow(img, cmap='gray')
plt.title("Your Handwritten Digit")
plt.axis('off')
plt.show()

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img_tensor = custom_transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    prediction = output.argmax(dim=1).item()

print(f"\nThe model thinks your digit is: {prediction}")