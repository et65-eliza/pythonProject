__author__ = 'tan_nguyen'

import os
import time
import numpy as np
import tensorflow as tf

# Load MNIST dataset from TensorFlow Keras datasets
from tensorflow.keras.datasets import mnist

# Load and preprocess the dataset (reshape and normalize)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build your network using Keras layers
def build_model():
    model = tf.keras.models.Sequential()

    # First convolutional layer
    model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same', name="conv1"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), name="maxpool1"))

    # Second convolutional layer
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same', name="conv2"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), name="maxpool2"))

    # Flatten and densely connected layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu', name="fc1"))

    # Dropout
    model.add(tf.keras.layers.Dropout(0.5))

    # Softmax output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax', name="output"))

    return model

def main():
    result_dir = './results/'  # directory where the results from the training are saved
    max_step = 5500  # maximum iterations
    batch_size = 50

    start_time = time.time()  # start timing

    # Build the model
    model = build_model()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Set up TensorBoard callback with additional logging for histograms and scalars
    log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='batch',
        profile_batch=0,
    )

    # Train the model with TensorBoard callback
    model.fit(x_train, y_train, epochs=5, batch_size=batch_size,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')

    stop_time = time.time()
    print('The training took %f seconds to complete' % (stop_time - start_time))

if __name__ == "__main__":
    main()
