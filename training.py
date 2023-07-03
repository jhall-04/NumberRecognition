import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers

# Load set of handwritten letters from MNIST
(train_img, train_lbl), (test_img, test_lbl) = tf.keras.datasets.mnist.load_data()

# Bring pixel values between range of 0 and 1
plt.imshow(train_img[0])
train_img = train_img / 255.0
test_img = test_img / 255.0

# Initialize model
model = tf.keras.Sequential()

# Implement layers in model
# Convolutional layers
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# Flatten layer
model.add(layers.Flatten())
# Dense layer
model.add(layers.Dense(units=64, activation='relu'))
# Final output layer
model.add(layers.Dense(units=10))

# Compile model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')

# Training hyperparameters
bucket_size = 600
num_epochs = 5

# Train model using dataset
hist = model.fit(train_img, train_lbl, batch_size=bucket_size, epochs=num_epochs, validation_data=(test_img, test_lbl))
# Check loss and accuracy
test_loss, test_accuracy = model.evaluate(test_img, test_lbl)

# Save model and output final loss and accuracy
model.save('number_model.h5', hist)
print(test_loss, test_accuracy)
print('Done :)')
