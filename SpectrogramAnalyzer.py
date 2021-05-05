import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.models import model_from_json
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class ConvNeuralNet:
    def __init__(self, json_path, h5_path, names_path):
        self.model = None
        self.class_names = None
        self.json_path = json_path
        self.h5_path = h5_path
        self.names_path = names_path

    def train_new(self, input_shape, data_set_path, batch_size, epochs):
        self.create_model(input_shape)
        self.train(input_shape, data_set_path, batch_size, epochs)

    def train_existing(self, input_shape, data_set_path, batch_size, epochs):
        self.load_model()
        self.train(input_shape, data_set_path, batch_size, epochs)

    def train(self, input_shape, data_set_path, batch_size, epochs):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_set_path,
            validation_split=0.25,
            subset="training",
            seed=123,
            image_size=(input_shape[0], input_shape[1]),
            batch_size=batch_size)
        self.class_names = train_ds.class_names
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_set_path,
            validation_split=0.25,
            subset="validation",
            seed=123,
            image_size=(input_shape[0], input_shape[1]),
            batch_size=batch_size)
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.model.summary()
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        self.save_model()

    def predict(self, image_path):
        self.load_model()
        img = keras.preprocessing.image.load_img(
            image_path, target_size=(170, 370)
        )
        single_image_arr = keras.preprocessing.image.img_to_array(img)
        single_image_arr = tf.expand_dims(single_image_arr, 0)
        predictions = self.model.predict(single_image_arr)
        score = tf.nn.softmax(predictions[0])
        # print( "This voice clip fits best in the category \"{}\" with a {:.2f} percent confidence.".format(self.class_names[np.argmax(score)], 100 * np.max(score)))
        return self.class_names[np.argmax(score)], score, self.class_names

    def start_model(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def create_model(self, input_shape):
        self.model = Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=input_shape),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(6)
        ])
        self.start_model()

    def load_model(self):
        json_file = open(self.json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        json_file = open(self.names_path, 'r')
        self.class_names = json.load(json_file)
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.h5_path)
        self.start_model()

    def save_model(self):
        model_json = self.model.to_json()
        with open(self.json_path, "w") as model_file:
            model_file.write(model_json)
        with open(self.names_path, "w") as names_file:
            json.dump(self.class_names, names_file)
        self.model.save_weights(self.h5_path)
