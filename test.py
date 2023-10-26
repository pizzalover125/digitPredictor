import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Load and preprocess the "test.png" image
img = load_img('test.png', color_mode='grayscale', target_size=(28, 28))
img_array = img_to_array(img)
img_array = img_array.reshape(1, 28, 28, 1)
img_array = img_array / 255.0  # Normalize the image data

# Make predictions
predictions = model.predict(img_array)

# Get the predicted digit (class) with the highest probability
predicted_digit = np.argmax(predictions)

print(f'Predicted digit: {predicted_digit}')
