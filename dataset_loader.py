import tensorflow as tf

# Set parameters
IMG_SIZE = 128
BATCH_SIZE = 32

# Function to load the dataset
def load_data(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    # Normalize the images to [0, 1] range
    def process(image, label):
        image = tf.cast(image/255., tf.float32)
        return image, label

    # Prefetch for performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(process)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = val_ds.map(process)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds

