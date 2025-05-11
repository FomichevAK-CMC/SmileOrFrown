print("Importing...")
import os, time, sys
from draw_box import DrawingApp
from PIL import Image, ImageDraw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # muting TF warnings

import keras.src.callbacks
from tensorflow.keras import datasets, layers, models
from keras.callbacks import EarlyStopping
import numpy as np


def load_images(image_file_set, shape):
    images = []
    for path in os.listdir(image_file_set):
        img = Image.open(image_file_set+path).convert('L').resize(shape)
        img = np.array(img).astype(np.float32) / 255.0
        images.append(img.flatten())
    return np.array(images)


def images_and_labels(classes_paths, shape):
    images = []
    labels = []

    for i in range(len(classes_paths)):
        cls_images = load_images(classes_paths[i], shape)
        images.extend(cls_images)
        labels.extend([i] * len(cls_images))
    return np.array(images), np.array(labels)


class SymbolRecognizer:
    def __init__(self, model, sym_num, img_shape):
        self.model = model
        self.sym_num = sym_num
        self.img_shape = img_shape

    def prepare_image(self, images):
        im = images.convert('L').resize(self.img_shape)
        tmp = (np.array(im) / 255.0).flatten().reshape(1, self.img_shape[0], self.img_shape[1], 1)
        return tmp

    def prepare_images(self, images):
        return images.reshape(len(images), self.img_shape[0], self.img_shape[1], 1)

    def train(self, train_images, train_labels, epochs=10, batch_size=8, validation_split=0.1):
        train_images = self.prepare_images(train_images)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=True ,callbacks=[early_stopping])

    def eval(self, test_images, test_labels):
        test_images = self.prepare_images(test_images)
        return self.model.evaluate(test_images, test_labels)

    def predict(self, image):
        return self.model.predict(self.prepare_image(image), verbose=0)

    @classmethod
    def from_layers(self, model_layers, img_shape):
        model_layers.insert(0, keras.Input(shape=(img_shape[0], img_shape[1], 1)))
        model = models.Sequential(model_layers)
        sym_num = model.get_config()["layers"][-1]["config"]["units"]
        return SymbolRecognizer(model, sym_num, img_shape)

    @classmethod
    def load_from_file(self, path='predictor.keras'):
        model = keras.models.load_model(path)
        inp = model.get_config()["layers"]
        img_shape = inp[0]['config']['batch_shape'][1:-1]
        num_sym = inp[-1]['config']['units']
        return SymbolRecognizer(model, num_sym, img_shape)

    def save(self, path='predictor.keras'):
        self.model.save(path)


def play_predict(model_file="predictor.json"):
    predictor = SymbolRecognizer.load_from_file(model_file)
    app = DrawingApp()
    while True:
        time.sleep(0.001)
        app.update()
        new = app.get_new_image()
        if new is None:
            continue
        syms = ["=)", "=("]
        prediction = predictor.predict(new)[0]
        higher = np.argmax(prediction)
        choice = syms[higher] if abs(0.5 - prediction[higher]) > 0.10 else "Unsure"
        app.set_text("Prediction: " + choice)
        print(choice, "- with the confidence of:", prediction[higher])


def train_model(save_to):
    model_layers = [
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ]
    predictor = SymbolRecognizer.from_layers(model_layers, (64, 64))

    images, labels = images_and_labels(['samples/sset_full/', 'samples/fset_full/'], predictor.img_shape)
    place = np.random.permutation(len(images))
    images = images[place]
    labels = labels[place]
    tr_im, tr_lb = images[:-500], labels[:-500]
    tt_im, tt_lb = images[-500:], labels[-500:]
    predictor.train(tr_im, tr_lb, epochs=1, batch_size=8, validation_split=0.15)
    predictor.save(save_to)
    acc = predictor.eval(tt_im, tt_lb)[1]
    print("Точность итоговой модели:", acc)


def main():
    print("Getting ready...")
    if '-t' in sys.argv:
        train_model("predictor64.keras")
    if '-p' in sys.argv:
        play_predict("predictor64.keras")


if __name__ == "__main__":
    main()
