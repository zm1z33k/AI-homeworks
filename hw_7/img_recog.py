import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import load_model

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Class names (digits)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Show image sample
def show_sample(s, c):
    plt.figure()
    plt.imshow(s, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.title(c)
    plt.show()

# path to saved model
model_path = "D:\\Unicorn Collage\\4.semestr\\AI_new\\hw_7\\mnist_digit_model.h5"

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading saved model...")
    model = load_model(model_path)

# if no model found, build and train a new one
else:
    print("Training model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),                   
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10)
    model.save(model_path)
    print("Model saved to disk.")

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# Predict on test set
predictions = model.predict(test_images)

# Prediction display function
def sample_predict(i, test_images, test_labels, predictions, class_names):
    s = test_images[i]
    c = class_names[test_labels[i]]
    print("Probabilities")
    print("------------")
    for j in range(len(class_names)):
        print(class_names[j], ":", np.round(predictions[i, j], 2))
    ind = np.argmax(predictions[i])
    print("------------")
    print("true class:", c, ", predicted class:", class_names[ind])

# Custom image preprocessing function
def preprocess_custom_image(image_path):

    # Load and convert image to grayscale
    img = Image.open(image_path).convert('L')

    # Invert if background is white
    if np.mean(img) > 127:
        img = ImageOps.invert(img)

    # Resize to 20x20
    img = img.resize((20, 20), Image.LANCZOS)

    # Paste into 28x28 black canvas, centered
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, (4, 4))

    # Convert to array and normalize
    img_array = np.array(new_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 28, 28)
    return img_array, new_img

# Predict on a custom image
img_path = 'D:\\Unicorn Collage\\4.semestr\\AI_new\\hw_7\\image.png'
img_array, preview_img = preprocess_custom_image(img_path)

# Predict the digit
custom_prediction = model.predict(img_array)
predicted_label = class_names[np.argmax(custom_prediction)]

# Show result
show_sample(preview_img, f"Predicted: {predicted_label}")
for i, prob in enumerate(custom_prediction[0]):
    print(f"{class_names[i]}: {prob * 100:.2f}%")
print("Predicted digit:", predicted_label)