import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load pretrained MobileNetV2 model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")

# Load and preprocess the image
img = Image.open("cat.jpg").resize((224, 224))
img_array = np.array(img)/255.0
img_array = img_array.astype(np.float32)
img_array = img_array[np.newaxis, ...]

# Predict
predictions = model(img_array)
predicted_id = np.argmax(predictions)

# Download labels
import requests
labels = requests.get("https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt").text.splitlines()

# Print result
print("Predicted:", labels[predicted_id])
