# import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the data
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words = 10000)

# Pad the sequences
max_length = max([len(s) for s in x_train])
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_length)

# Build the model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.SimpleRNN(32))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs = 10, validation_data = (x_val, y_val))
