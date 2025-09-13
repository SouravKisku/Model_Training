import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# ✅ Same IMG_SIZE as training
IMG_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model("model.h5")

# ✅ Load class indices from training (saved earlier)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping {0: 'cat', 1: 'dog'}
class_labels = {v: k for k, v in class_indices.items()}

# Load and preprocess test image
img_path = "test.jpg"   # apna test image path yahan do
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_labels[int(prediction[0][0] > 0.5)]

print(f"Predicted: {predicted_class}")
