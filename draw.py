# Import libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# Global variable
global lasx, lasy

# Import data
(train_img, train_lbl), (test_img, test_lbl) = tf.keras.datasets.mnist.load_data()

# Load model
model = tf.keras.models.load_model('number_model.h5')

# Iterates through files converting PNG to numpy array then guesses the number written
# NeuralNine https://www.youtube.com/watch?v=PvaMNbkIXmY
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"This digit is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    image_number += 1
