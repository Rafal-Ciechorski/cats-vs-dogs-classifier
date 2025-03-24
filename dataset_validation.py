import os
from PIL import Image

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