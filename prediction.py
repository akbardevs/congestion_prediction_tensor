import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    # Load the image, resizing it to the expected input size of the model
    img = image.load_img(img_path, target_size=(720, 1280))
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array /= 255.0  # Scale the pixel values to [0, 1]
    return img_array

# Function to predict congestion using the loaded model
def predict_congestion(model, img_array):
    prediction = model.predict(img_array)
    return prediction

# Assuming the model is stored at 'path_to_your_model.h5'
model_path = './model_training.h5'
model = tf.keras.models.load_model(model_path)

# Replace 'path_to_your_image.jpg' with the path to your test image
img_path = './image_prediction/sample1.jpg'
img_to_predict = load_and_preprocess_image(img_path)

# Use the loaded model to predict the congestion
congestion_prediction = predict_congestion(model, img_to_predict)

# Output the predicted congestion level
print("Predicted congestion level:", congestion_prediction[0][0])