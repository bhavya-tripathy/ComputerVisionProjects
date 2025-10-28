import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np

# --- Configuration ---
IMG_HEIGHT = 28
IMG_WIDTH = 28
BATCH_SIZE = 32
NUM_CLASSES = 26 # A-Z
EPOCHS = 50
TRAIN_DATA_PATH = 'data/sign_mnist_train.csv'
TEST_DATA_PATH = 'data/sign_mnist_test.csv'
MODEL_SAVE_PATH = 'sign_language_model.h5'

def load_data(train_path, test_path):
    """
    Loads the MNIST sign language dataset from CSV files.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df['label'].values
    y_test = test_df['label'].values

    X_train = train_df.drop('label', axis=1).values
    X_test = test_df.drop('label', axis=1).values
    X_train = X_train.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    X_test = X_test.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)

    return X_train, y_train, X_test, y_test

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    print("Data loaded.")
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False, 
        fill_mode='nearest'
    )

    X_train_normalized = X_train / 255.0
    X_test_normalized = X_test / 255.0

    print("Creating model...")
    model = create_model()
    model.summary()

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    print("Training model...")
    history = model.fit(
        datagen.flow(X_train_normalized, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test_normalized, y_test),
        callbacks=[checkpoint, early_stopping]
    )

    print(f"Model trained and saved to {MODEL_SAVE_PATH}")

    loss, accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

if __name__ == "__main__":
    # Note: To run this, you need the Sign Language MNIST dataset.
    # You can download it from Kaggle:
    # https://www.kaggle.com/datasets/datamunge/sign-language-mnist
    # Make sure to place sign_mnist_train.csv and sign_mnist_test.csv in a 'data' folder.
    main()
