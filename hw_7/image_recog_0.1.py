import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import load_model
import random

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Class names (digits)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Randomly select 10% of the test dataset
num_samples = int(0.1 * len(test_images))
random_indices = random.sample(range(len(test_images)), num_samples)
saved_images = [test_images[i] for i in random_indices]
saved_labels = [test_labels[i] for i in random_indices]

# Show image sample
def show_sample(s, c):
    plt.figure()
    plt.imshow(s, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.title(c)
    plt.show()

# Path to saved model
model_path = "D:\\Unicorn Collage\\4.semestr\\AI_new\\hw_7\\mnist_digit_model.h5"

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading saved model...")
    model = load_model(model_path)
else:
    print("Training model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),                   
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    model.save(model_path)
    print("Model saved to disk.")

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# Predict and evaluate accuracy on 10% of dataset samples
correct = 0
for i, saved_image in enumerate(saved_images):

    # Show the image sample
    img_array = np.expand_dims(saved_image, axis=0)
    custom_prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(custom_prediction)
    predicted_label = class_names[predicted_index]
    true_label = saved_labels[i]

    print(f"Sample {i + 1}:")
    print(f"True Label: {class_names[true_label]}, Predicted Label: {predicted_label}")
    for j, prob in enumerate(custom_prediction[0]):
        print(f"{class_names[j]}: {prob * 100:.2f}%")
    print("------------")

    if predicted_index == true_label:
        correct += 1

average_accuracy = correct / num_samples
print(f"\nAverage accuracy on {num_samples} random samples: {average_accuracy * 100:.2f}%")
