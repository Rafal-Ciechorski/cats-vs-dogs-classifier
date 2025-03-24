import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from dataset_loader import load_data  # Importing the dataset loader

# Parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = r"D:\Projects\CatsAndDogsDataSet\Models\cat_dog_cnn.keras"
DATA_DIR = r"D:\Projects\CatsAndDogsDataSet\PetImages"

# Load dataset
train_ds, val_ds = load_data(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# CNN model
def create_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (Cat vs Dog)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_cnn_model()

# Train the model
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# Save the model
model.save(MODEL_SAVE_PATH)

# Plot training results (Accuracy & Loss)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()