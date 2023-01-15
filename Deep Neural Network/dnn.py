# import libraries
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
data = np.random.rand(1000, 100)
labels = np.random.randint(0, 2, (1000, 1))

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3)

# Build the model
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs = 20, batch_size = 32, validation_data = (X_test, y_test))
