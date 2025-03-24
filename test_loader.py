from dataset_loader import load_data

# Set parameters
DATA_DIR = r"D:\Projects\CatsAndDogsDataSet\PetImages"
IMG_SIZE = 128
BATCH_SIZE = 32

# Test loading the dataset
train_ds, val_ds = load_data(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# Check the dataset
print("Training dataset batches:", len(train_ds))
print("Validation dataset batches:", len(val_ds))

# Check the shape of a batch of images (should be (batch_size, IMG_SIZE, IMG_SIZE, 3))
for images, labels in train_ds.take(1):
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
