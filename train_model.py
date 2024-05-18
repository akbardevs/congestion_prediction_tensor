# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

# # Set up directories for training and validation datasets
# train_dir = './train'
# validation_dir = './valid'

# # Set up image generators for data augmentation and rescaling
# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)

# # Prepare data generators for training and validation sets
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(720, 1280),  # Adjust to the size of your input images
#     batch_size=20,
#     class_mode='binary')  # Use 'categorical' for more than two classes

# validation_generator = val_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(720, 1280),
#     batch_size=20,
#     class_mode='binary')

# # Define the CNN model architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(720, 1280, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Print model summary to check the structure
# model.summary()
# model.save('model_training.h5')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Reducing batch size to lessen memory usage
batch_size = 10  # Reduced from 20

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'path_to_train_directory',
    target_size=(720, 1280),
    batch_size=batch_size,
    class_mode='binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(720, 1280, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=100, epochs=10)  # Adjust steps_per_epoch accordingly

model.save('path_to_your_model', save_format='tf')

# Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=100,  # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
#     epochs=10,
#     validation_data=validation_generator,
#     validation_steps=50  # Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
# )

# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'], 'r', label='Training accuracy')
# plt.plot(history.history['val_accuracy'], 'b', label='Validation accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()

# # Plot training & validation loss values
# plt.figure()
# plt.plot(history.history['loss'], 'r', label='Training Loss')
# plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
