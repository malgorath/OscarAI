import tensorflow as tf

# Define a simple example model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model (for demonstration purposes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Generate some dummy data for training (you should replace this with your data)
import numpy as np
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
y_train = y_train.astype(np.int32)

# Train the model (for demonstration purposes)
model.fit(x_train, y_train, epochs=5)

# Save the trained model to a file
model.save('model.h5')
