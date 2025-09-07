A Convolutional Neural Network (CNN) built with PyTorch that recognizes any handwritten digit after being trained on a large dataset of handwritten numbers. You can draw a number, and the model will predict what digit it is (0–9).

1. Download essential libraries, by going to the console and pasting "pip install torch torchvision matplotlib pillow"
2. Run NML.py
3. It will either train a new model or use the mnist_cnn.pth file that was previously trained on a prior run of the code.
4. Draw a Digit using the built in UI.
6. It will convert your drawing into a image and it will preprocess the image to make to look similar to other images from the dataset (greyscaled and pixelized).
8. Then it will predict what number the handwritten digit is based on the weights it was trained on!

