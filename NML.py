import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
import os
import numpy as np
import tkinter as tk


# centers image for preprocessing
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

# Function for inputting your own image file instead of drawing one
def PredOwnImg(path):
    img_path = "HW-Nums//digit1.PNG"  # put image path here
    img = Image.open(path).convert('L')
    img = ImageOps.invert(img)
    img = center_image(img)

    # Show the preprocessed image
    plt.clf()  # clear the current figure
    plt.gcf().set_size_inches(3, 3)  # resize current figure (width=3", height=3")
    plt.imshow(img, cmap='gray')
    plt.title("Preprocessed Digit")
    plt.axis('off')
    plt.show(block=False)

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = custom_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax(dim=1).item()

    print(f"\nThe model thinks your digit is: {prediction}")


# Function that lets you draw your own digit
def draw():
    window_size = 400
    brush_size = 6

    root = tk.Tk()
    root.title("Draw a Digit")
    canvas = tk.Canvas(root, width=window_size, height=window_size, bg="white")
    canvas.pack()

    # --- Create PIL image to mirror canvas ---
    canvas_image = Image.new("L", (window_size, window_size), 255)  # white background
    draw_image = ImageDraw.Draw(canvas_image)

    # --- Prediction label ---
    pred_label = tk.Label(root, text="", font=("Arial", 30))

    def paint(event):
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        draw_image.ellipse([x1, y1, x2, y2], fill=0)  # draw on PIL image

    def clear():
        canvas.delete("all")
        draw_image.rectangle([0, 0, window_size, window_size], fill=255)
        pred_label.config(text="")

    def predict():
        os.makedirs("drawings", exist_ok=True)
        save_path = "drawings/drawn_digit.png"

        img = canvas_image
        img.save(save_path)

        # preprocess image
        img_pre = ImageOps.invert(canvas_image)
        img_pre = center_image(img_pre)

        # Show the preprocessed image
        plt.clf()  # clear the current figure
        plt.gcf().set_size_inches(3, 3)  # resize current figure (width=3", height=3")
        plt.imshow(img_pre, cmap='gray')
        plt.title("Preprocessed Digit")
        plt.axis('off')
        plt.show(block=False)

        # use model to predict what digit it is
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = custom_transform(img_pre).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.argmax(dim=1).item()

        # Update the label
        pred_label.config(text=f"Digit = {prediction}")

    canvas.bind("<B1-Motion>", paint)

    btn_frame = tk.Frame(root)
    btn_frame.pack()

    tk.Button(btn_frame, text="Clear", command=clear, width=12, height=1, font=("Arial", 14)).pack(side="left", padx=10, pady=4)
    tk.Button(btn_frame, text="Predict", command=predict, width=12, height=1, font=("Arial", 14)).pack(side="left", padx=10, pady=4)
    pred_label.pack(pady=10)

    root.mainloop()

# Code starts running here
if __name__ == "__main__":
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
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
        model.eval()
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        # Only load training data if model doesn't exist
        train_data = datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
        train_loader = DataLoader(
            train_data,
            batch_size=256,  # try 128, 256, 512, etc.
            shuffle=True,
            num_workers=4,  # let CPU preload batches
            pin_memory=True  # faster transfer to GPU
        )

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

    # predict image that you drew
    model.eval()
    draw()