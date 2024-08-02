import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import arff
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_preprocess_data(ECG5000_TEST):
    with open(ECG5000_TEST) as f:
        data = arff.load(f)
    
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)  # Convert labels to integers if needed
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_test, y_train, y_test

# Define the model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()

# Plot predictions vs true labels
def plot_predictions(y_test, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='True Labels')
    plt.plot(predictions, label='Predicted Labels')
    plt.xlabel('Samples')
    plt.ylabel('Labels')
    plt.legend()
    plt.title('True Labels vs Predicted Labels')
    plt.show()

# Main function
def main():
    choice = input("Select data type (ECG/EEG): ").strip().lower()
    
    if choice == 'ecg':
        file_path = 'ECG5000_TEST.arff'  # Replace with actual file path
    elif choice == 'eeg':
        file_path = 'EEG_TEST.arff'  # Replace with actual file path
    else:
        print("Invalid choice. Please select either 'ECG' or 'EEG'.")
        return
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    
    # Build and train the model
    model = build_model((X_train.shape[1], 1))
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    # Save the model
    model_file_name = f'{choice}_model.h5'
    model.save(model_file_name)
    print(f'Model saved as {model_file_name}')
    
    # Load the model for future predictions
    model = tf.keras.models.load_model(model_file_name)
    
    # Predict anomalies
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)  # Binarize the predictions
    
    # Plot predictions vs true labels
    plot_predictions(y_test, predictions)

if __name__ == "__main__":
    main()
