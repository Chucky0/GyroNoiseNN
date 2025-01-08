import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class ModelWrapper:
    def __init__(self, model_creation_func, input_shape, model_name):
        self.model = model_creation_func(input_shape)
        self.model_name = model_name
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, checkpoint_file=None, start_epoch=0):
        """Тренує модель."""

        X_train = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        callbacks = []

        if checkpoint_file:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath= os.path.join("model_weights", f"{checkpoint_file}_{self.model_name}"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint_callback)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            initial_epoch=start_epoch,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        return history

    def predict(self, X_test):
        """Робить передбачення на тестових даних."""
        X_test = self.scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        y_pred = self.model.predict(X_test).flatten()
        y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred

    def save(self, filepath):
        """Зберігає модель у форматі TensorFlow Lite."""
        # Конвертуємо модель у TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        # Зберігаємо конвертовану модель
        with open(f"{filepath}.tflite", "wb") as f:
            f.write(tflite_model)

    def load(self, filepath):
        """Завантажує модель."""
        self.model.load_weights(filepath)

def create_rnn_model(input_shape):
    """Створює модель RNN."""
    model = keras.Sequential([
        keras.layers.SimpleRNN(50, activation='relu', return_sequences=True, input_shape=input_shape),
        keras.layers.SimpleRNN(50, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_lstm_model(input_shape):
    """Створює модель LSTM."""
    model = keras.Sequential([
        keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(50, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_gru_model(input_shape):
    """Створює модель GRU."""
    model = keras.Sequential([
        keras.layers.GRU(50, activation='relu', return_sequences=True, input_shape=input_shape),
        keras.layers.GRU(50, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_cnn_model(input_shape):
    """Створює модель CNN."""
    model = keras.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_cnn_lstm_model(input_shape):
    """Створює гібридну модель CNN-LSTM."""
    model = keras.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.LSTM(50, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model