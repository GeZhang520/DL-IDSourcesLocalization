import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


class IDSourceEstimator:
    def __init__(self, input_shape, angle_units, spread_units):
        self.input_shape = input_shape
        self.angle_units = angle_units
        self.spread_units = spread_units
        self.model = self.createModel()

    def createModel(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        # --------- auto-encoder --------
        x = Dense(units=500, activation=tf.nn.relu)(x)
        x = Dense(units=200, activation=tf.nn.relu)(x)
        x = Dense(units=80, activation=tf.nn.relu)(x)
        x = Dense(units=200, activation=tf.nn.relu)(x)
        x = Dense(units=500, activation=tf.nn.relu)(x)
        # -------------------------------
        # ----- parallel sub-networks ---
        a1 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu)(x)
        a2 = tf.keras.layers.Dense(units=60, activation=tf.nn.relu)(a1)

        s1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
        s2 = tf.keras.layers.Dense(units=60, activation=tf.nn.relu)(s1)
        # -------------------------------
        x = tf.keras.layers.concatenate([a2, s2])
        # ------ Linear layers ----------
        l1 = tf.keras.layers.Dense(units=self.angle_units, name="angle_output")(x)
        l2 = tf.keras.layers.Dense(units=self.spread_units, name="spread_output")(x)
        # -------------------------------
        self.model = tf.keras.Model(inputs=inputs, outputs=[l1, l2])

        self.model.compile(
            # optimizer and learning rate
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            # --- loss function, Weighted-MSE ---
            loss={'angle_output': [tf.keras.losses.MSE],
                  'spread_output': [tf.keras.losses.MSE]},
            loss_weights={"angle_output": 1.0, "spread_output": 2.0}
            # -----------------------------------
        )
        return self.model

    def fit(self, train_inputs, train_angle_labels, train_spread_labels):
        history = self.model.fit(
            x=train_inputs,
            y={'angle_output': train_angle_labels,
               'spread_output': train_spread_labels},
            # 80% for train and 20% for validation
            validation_split=0.2,
            batch_size=100,
            epochs=1000,
            verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss')]
        )
        return history

    def loadWeights(self, file_path):
        self.model.load_weights(filepath=file_path)

    def saveWeights(self, file_path):
        self.model.save(filepath=file_path)

    def plotModelStructure(self, file_path):
        tf.keras.utils.plot_model(model=self.model, to_file=file_path, show_shapes=True)

    def predict(self, inputs, signal_num):
        x = np.expand_dims(inputs, axis=0)
        angle_pred, spread_pred = self.model.predict(x)
        return angle_pred[0, 0:signal_num], spread_pred[0, 0:signal_num]

    def predictOnBatch(self, inputs, signal_num):
        angle_pred, spread_pred = self.model.predict_on_batch(inputs)
        return angle_pred[:, 0:signal_num], spread_pred[:, 0:signal_num]
