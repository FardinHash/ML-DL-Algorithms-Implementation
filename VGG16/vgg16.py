# import libraries
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
        'data/val',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'binary')

# Load the pre-trained model
conv_base = VGG16(weights = 'imagenet', include_top=False, input_shape = (224, 224, 3))

# Freeze the convolutional base
conv_base.trainable = False

# Build the model
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch = 100,
                    epochs = 10,
                    validation_data=validation_generator,
                    validation_steps = 50)
