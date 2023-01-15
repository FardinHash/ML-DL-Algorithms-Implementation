# import libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Generate input sequence
sequence = np.random.rand(100,1)

# Split the sequence into input and output data
X, y = sequence[:-1], sequence[-1]

# Reshape the input data for the LSTM network
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(10, input_shape = (X.shape[1], X.shape[2])))
model.add(Dense(1))

# Compile the model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Train the model
model.fit(X, y, epochs=100, batch_size = 1, verbose = 2)
