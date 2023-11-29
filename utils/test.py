import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2

# To be able to run tests locally - replace with the local path to dataset
ship_dir = '/kaggle/input/airbus-ship-detection'
test_image_dir = os.path.join(ship_dir, 'test_v2')

unet_like = keras.models.load_model('/kaggle/working/unet_like.keras')

def load_images(image):
    path = test_image_dir + '/' + image
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    return image

test_images_pixels = []
test_images_name = os.listdir(test_image_dir)[10:12]

for image_name in test_images_name:
    test_images_pixels.append(load_images(image_name))
    
test_images_tensor = tf.convert_to_tensor(test_images_pixels)

pred = unet_like.predict(test_images_tensor)
pred = np.squeeze(pred)

pred = np.where(pred < 0.02, 0.0, 1.0)

# To decrease noises Gaussian Blur is used
blur0 = cv2.GaussianBlur(pred[0], (7,7), cv2.BORDER_DEFAULT)
blur1 = cv2.GaussianBlur(pred[1], (7,7), cv2.BORDER_DEFAULT)

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(2, 2, 1)
plt.imshow(test_images_pixels[0]*255)
plt.title("Image")
fig.add_subplot(2, 2, 2)
plt.imshow(blur0)
plt.title("Prediction")
fig.add_subplot(2, 2, 3)
plt.imshow(test_images_pixels[1]*255)
plt.title("Image")
fig.add_subplot(2, 2, 4)
plt.imshow(blur1)
plt.title("Prediction")
plt.show()