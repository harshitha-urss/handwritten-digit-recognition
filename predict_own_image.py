import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 1. Load trained model
model = keras.models.load_model("mnist_cnn.keras")

def predict_own_image(path, threshold=0.7):
    if not os.path.exists(path):
        print("File not found:", path)
        return

    # 2. Open and preprocess image
    img = Image.open(path).convert("L")      # grayscale
    img = img.resize((28, 28))               # 28x28
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # shape (1, 28, 28, 1)

    # 3. Predict
    probs = model.predict(arr)
    pred_class = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs))

    if confidence < threshold:
        print(f"Cannot be recognised (max confidence = {confidence:.2f})")
        plt.imshow(arr[0, :, :, 0], cmap="gray")
        plt.title("Cannot be recognised")
    else:
        print(f"Predicted digit: {pred_class} (confidence = {confidence:.2f})")
        plt.imshow(arr[0, :, :, 0], cmap="gray")
        plt.title(f"Predicted: {pred_class} ({confidence:.2f})")

    plt.axis("off")
    plt.show()

# change this filename to your image
predict_own_image("my_digit1.png")

