import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import numpy as np

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential()
model.add(Input(shape=(28, 28, 1))),
layers.Flatten(),
model.add(layers.Conv2D(32, (3, 3), activation='relu',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Visualize training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Make predictions
predictions = model.predict(x_test)

# Get the predicted classes
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Class names for Fashion MNIST
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Visualize predictions with colored text
def plot_predictions(images, true_labels, predicted_labels, start_index, end_index):
    plt.figure(figsize=(15, 5))
    for i in range(start_index, end_index):
        plt.subplot(4, 5, i - start_index + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        true_name = class_names[true_label]
        predicted_name = class_names[predicted_label]
        color = 'green' if true_label == predicted_label else 'red'
        plt.title(f'True: {true_name}\nPred: {predicted_name}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Specify the range of predictions to visualize
start_index = 5405  # Change this to your desired start index
end_index = 5425  # Change this to your desired end index

# Plot predictions for the specified range
plot_predictions(x_test, true_classes, predicted_classes, start_index, end_index)
