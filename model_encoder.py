import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class BaseModel:
    def __init__(self, window_size, num_sensors):
        self.window_size = window_size
        self.num_sensors = num_sensors  # Тепер це кількість вхідних сенсорів
        self.model = None

    def build(self):
        raise NotImplementedError

    def build_with_params(self, **params):
        raise NotImplementedError

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=0, model_filepath="model"):
        self.model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

        checkpoint = ModelCheckpoint(f"{model_filepath}.keras", monitor='val_loss', save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

        # Ініціалізуємо history як None
        history = None

        try:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[checkpoint, early_stopping],
                verbose=verbose
            )
        except Exception as e:
            print(f"Error during model training: {e}")

        # Повертаємо history або None, якщо виникла помилка
        return history

    def predict(self, X):
        # Передбачення тільки на основі вхідних даних (без Encoder Speed)
        return self.model.predict(X, verbose=0)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)


class LSTMModel(BaseModel):
    def build(self):
        self.build_with_params(units=50, learning_rate=0.001)

    def build_with_params(self, units, learning_rate):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)),
            LSTM(units, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(1)  # Вихідний шар передбачає одне значення (Encoder Speed)
        ])
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')


class CNNModel(BaseModel):
    def build(self):
        self.build_with_params(filters=64, kernel_size=3, dense_units=50, learning_rate=0.001)

    def build_with_params(self, filters, kernel_size, dense_units, learning_rate):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)),
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(dense_units, activation='relu'),
            Dense(1)  # Вихідний шар передбачає одне значення (Encoder Speed)
        ])
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')


class CNNLSTMModel(BaseModel):
    def build(self):
        self.build_with_params(conv_filters=64, conv_kernel_size=3, lstm_units=50, learning_rate=0.001)

    def build_with_params(self, conv_filters, conv_kernel_size, lstm_units, learning_rate):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)),
            Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(lstm_units, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(1)  # Вихідний шар передбачає одне значення (Encoder Speed)
        ])
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')


class TCNModel(BaseModel):
    def __init__(self, window_size, num_sensors):
        super().__init__(window_size, num_sensors)
        from tensorflow_addons.layers import TCN

        self.TCN = TCN

    def build(self):
        self.build_with_params(filters=64, kernel_size=3, dense_units=50, learning_rate=0.001)

    def build_with_params(self, filters, kernel_size, dense_units, learning_rate):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)),
            self.TCN(nb_filters=filters, kernel_size=kernel_size, dilations=[1, 2, 4, 8], padding='causal',
                     use_skip_connections=True, return_sequences=False, activation='relu'),
            Dense(dense_units, activation='relu'),
            Dense(1)  # Вихідний шар передбачає одне значення (Encoder Speed)
        ])
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')


class GRUModel(BaseModel):
    def build(self):
        self.build_with_params(units=50, learning_rate=0.001)

    def build_with_params(self, units, learning_rate):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)),
            GRU(units, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(1)  # Вихідний шар передбачає одне значення (Encoder Speed)
        ])
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')