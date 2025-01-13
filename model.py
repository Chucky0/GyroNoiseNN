import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, GRU, Input # Додано Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class BaseModel:
    def __init__(self, window_size, num_sensors, units=64, filters=64, kernel_size=3, dense_units=64, lstm_units=64, learning_rate = 0.0001):
        self.window_size = window_size
        self.num_sensors = num_sensors
        self.units = units
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = None

    def build(self):
        raise NotImplementedError

    def build_with_params(self, units=None, filters=None, kernel_size=None, dense_units=None, lstm_units=None, learning_rate=None):
        # Оновлення атрибутів на основі переданих параметрів
        if units is not None:
            self.units = units
        if filters is not None:
            self.filters = filters
        if kernel_size is not None:
            self.kernel_size = kernel_size
        if dense_units is not None:
            self.dense_units = dense_units
        if lstm_units is not None:
            self.lstm_units = lstm_units
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.build()

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_filepath="model", optimizer=Adam()):
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        checkpoint = ModelCheckpoint(f"{model_filepath}.keras", monitor='val_loss', save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )

        return history

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)


class LSTMModel(BaseModel):
    def build(self):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)), # Виправлено: Input тепер імпортовано
            LSTM(self.units, activation='relu'),
            Dense(1)
        ])

class CNNModel(BaseModel):
    def build(self):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)), # Виправлено: Input тепер імпортовано
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(self.dense_units, activation='relu'),
            Dense(1)
        ])

class CNNLSTMModel(BaseModel):
    def build(self):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)), # Виправлено: Input тепер імпортовано
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(self.lstm_units, activation='relu'),
            Dense(1)
        ])

class TCNModel(BaseModel):
    def build(self):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)), # Виправлено: Input тепер імпортовано
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', padding='causal'),
            tf.keras.layers.BatchNormalization(),
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', dilation_rate=2, padding='causal'),
            tf.keras.layers.BatchNormalization(),
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', dilation_rate=4, padding='causal'),
            tf.keras.layers.BatchNormalization(),
            Flatten(),
            Dense(self.dense_units, activation='relu'),
            Dense(1)
        ])

class GRUModel(BaseModel):
    def build(self):
        self.model = Sequential([
            Input(shape=(self.window_size, self.num_sensors)), # Виправлено: Input тепер імпортовано
            GRU(self.units, activation='relu'),
            Dense(1)
        ])