import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import numpy as np

#check to see if it can use GPU
print("PyTorch CUDA version:", torch.version.cuda)  # Should be close to 12.1, 12.0, or 11.x
print("CUDA available:", torch.cuda.is_available())  # Should be True
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# --- Check and set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. Load MNIST data (train and test)
mnist_transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 2. Define a CNN model for MNIST
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (N,1,28,28) -> (N,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> (N,32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# -> (N,64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)                              # -> (N,64,7,7)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNNNet().to(device)

# 3. Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

MODEL_PATH = "mnist_cnn.pth"

# 4. Load model or train and save
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded model weights from {MODEL_PATH}")
else:
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

# 5. Show model predictions on test set
model.eval()
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

example_data = example_data.to(device)
with torch.no_grad():
    output = model(example_data)

fig = plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    # Move tensor back to CPU and detach for plotting
    img = example_data[i].cpu().numpy().squeeze()
    plt.imshow(img, cmap='gray', interpolation='none')
    pred_label = output[i].argmax().item()
    true_label = example_targets[i].item()
    plt.title(f"Pred: {pred_label}, True: {true_label}")
    plt.axis('off')
plt.show()

# 6. Predict your own handwritten digit
def center_image(img):
    np_img = np.array(img)
    np_img[np_img < 20] = 0  # Optional noise cleanup
    coords = np.argwhere(np_img > 0)
    if coords.size == 0:
        return img.resize((28, 28))  # fallback

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img.crop((x0, y0, x1, y1))

    # Resize to fit into 20x20 box (as in MNIST)
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Create a blank 28x28 image and paste centered
    centered_img = Image.new('L', (28, 28), color=0)
    x_offset = (28 - cropped.width) // 2
    y_offset = (28 - cropped.height) // 2
    centered_img.paste(cropped, (x_offset, y_offset))

    return centered_img

img_path = "my_digit6.png"  # Change this to your image path
img = Image.open(img_path).convert('L')         # Convert to grayscale
img = ImageOps.invert(img)                      # Invert: MNIST is white-on-black
img = center_image(img)                         # Center the digit

# Show the image (optional)
plt.imshow(img, cmap='gray')
plt.title("Your Handwritten Digit (Preprocessed and Centered)")
plt.axis('off')
plt.show()

# Transform to match MNIST normalization
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img_tensor = custom_transform(img).unsqueeze(0).to(device)  # Shape: (1, 1, 28, 28), move to device

# Predict
model.eval()
with torch.no_grad():
    output = model(img_tensor)
    prediction = output.argmax(dim=1).item()

print(f"\n🧠 The model thinks your digit is: {prediction}")