import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class BaseModel:
    def __init__(self, window_size, num_sensors):
        self.window_size = window_size
        self.num_sensors = num_sensors
        self.model = None

    def build(self):
        raise NotImplementedError

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_filepath="model.h5"):
        self.model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

        checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True, mode='min')
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
            LSTM(50, activation='relu', input_shape=(self.window_size, self.num_sensors)),
            Dense(1)
        ])


class CNNModel(BaseModel):
    def build(self):
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.window_size, self.num_sensors)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])


class CNNLSTMModel(BaseModel):
    def build(self):
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.window_size, self.num_sensors)),
            MaxPooling1D(pool_size=2),
            LSTM(50, activation='relu'),
            Dense(1)
        ])


class TCNModel(BaseModel):
    def build(self):
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.window_size, self.num_sensors),
                   padding='causal'),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=3, activation='relu', dilation_rate=2, padding='causal'),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=3, activation='relu', dilation_rate=4, padding='causal'),
            BatchNormalization(),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])


class GRUModel(BaseModel):
    def build(self):
        self.model = Sequential([
            GRU(50, activation='relu', input_shape=(self.window_size, self.num_sensors)),
            Dense(1)
        ])