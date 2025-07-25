import tensorflow as tf
import urllib.request
import zipfile
import os

def download_data():
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    if not os.path.exists("cats_and_dogs_filtered.zip"):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, "cats_and_dogs_filtered.zip")
    
    if not os.path.exists("cats_and_dogs_filtered"):
        print("Extracting...")
        with zipfile.ZipFile("cats_and_dogs_filtered.zip", 'r') as zip_ref:
            zip_ref.extractall()

def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def main():
    download_data()
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "cats_and_dogs_filtered/train",
        image_size=(224, 224),
        batch_size=32
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "cats_and_dogs_filtered/validation",
        image_size=(224, 224),
        batch_size=32
    )
    
    train_ds = train_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))
    
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_ds, validation_data=val_ds, epochs=5)
    
    model.save("model")
    print("Done!")

if __name__ == "__main__":
    main()
