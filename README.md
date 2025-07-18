A simple Convolutional Neural Network (CNN) built with PyTorch that recognizes handwritten digits from the MNIST dataset. You can draw a number, and the model will predict what digit it is (0–9).

1. Download essential libraries, by going to the console and pasting "pip install torch torchvision matplotlib pillow"
2. Run NML.py
3. It will either train a new model or use the mnist_cnn.pth file that was already trained on a previous run of the code
4. Then it will read a handwritten digit image
5. It will convert the image to look similar to other images from the dataset (greyscaled and pixelized)
6. Then it will predict what number the handwritten digit is based on the weights it was trained on!

