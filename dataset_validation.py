import os
from PIL import Image
import tensorflow as tf


# Dataset path
dataset_path = r"D:\Projects\CatsAndDogsDataSet\PetImages"

# Function to re-save all images in database
def process_all_images(dataset_path):
    processed_count = 0
    deleted_count = 0

    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    img.save(file_path, format="JPEG", quality=95)
                processed_count += 1
            except:
                os.remove(file_path)  # Delete image beyond repair
                deleted_count += 1

    print(f"Processed {processed_count} images.")
    print(f"Removed {deleted_count} irreparable images.")

process_all_images(dataset_path)

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(r"D:\Projects\CatsAndDogsDataSet\PetImages", folder_name)
    for fname in os.listdir(folder_path):
        print(fname) # <---- add this
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(224, 224),  # Resize images
    batch_size=32
)

# Iterate through the dataset to check for errors
for images, labels in dataset:
    pass

print("Dataset successfully loaded without errors!")