import PIL.Image
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import string
from keras import backend as K


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_WIDTH = 256
TARGET_HEIGHT = 32
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)
EPOCHS = 10
ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase + "!?.,;:-'\"%$&/()=*#+|"
SPLIT_FACTOR = 0.01


def main():
    # filename: [ok, width, height, label]
    descriptions = load_descriptions()
    images = load_image_names()
    np.random.shuffle(images)
    split_index = int(len(images) * SPLIT_FACTOR)

    model = train(images, split_index, descriptions)
    #test(images, split_index, descriptions, model)


def test(images, split_index, descriptions, model):
    test_images = images[split_index:]
    test_labels = [descriptions[os.path.splitext(os.path.basename(image))[0]][3] for image in test_images]
    test_labels = np.array([np.array([ALPHABET.index(c) for _, c in enumerate(label)]) for label in test_labels])
    test_images = np.array([load_image(image) for image in test_images])


def train(images, split_index, descriptions):
    training_images = images[:split_index]
    training_labels = [descriptions[os.path.splitext(os.path.basename(image))[0]][3] for image in training_images]
    training_labels = np.array([0.0 for label in training_labels])
    training_images = np.array([load_image(image) for image in training_images])
    print(training_labels[0])
    print(training_images[0])

    model = train_model(training_images, training_labels)
    return model


def train_model(training_images, training_labels):
    kernel = (3, 3)
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(TARGET_WIDTH, TARGET_HEIGHT)),
        keras.layers.Reshape(target_shape=(TARGET_WIDTH, TARGET_HEIGHT, 1)),
        keras.layers.Conv2D(filters=32, kernel_size=kernel, activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=32, kernel_size=kernel, activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=32, kernel_size=kernel, activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #keras.layers.Dropout(0.25),
        #keras.layers.Flatten(),
        keras.layers.Reshape(target_shape=(TARGET_WIDTH//4, TARGET_HEIGHT//4*32)),
        keras.layers.Dense(TARGET_WIDTH, activation=tf.nn.relu),
        keras.layers.GRU(512, return_sequences=True),
        keras.layers.GRU(512, return_sequences=True, go_backwards=True),
        keras.layers.GRU(512, return_sequences=True),
        keras.layers.GRU(512, return_sequences=True, go_backwards=True),
        keras.layers.Dense(len(ALPHABET), activation=tf.nn.softmax)
    ])

    model.summmary()

    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the digit classification model
    model.fit(training_images, training_labels, epochs=EPOCHS)
    return model


def ctc(prediction, labels, input_length, label_length):
    K.ctc_decode(labels, prediction, input_length, label_length)


def load_image(path):
    image = Image.open(path)
    image.thumbnail(TARGET_SIZE)
    image_array = np.asarray(image)
    image_array = image_array / 255.0

    padding_width = TARGET_WIDTH - len(image_array[0])
    padding_height = TARGET_HEIGHT - len(image_array)
    width_padding_index = random.randint(0, padding_width)
    height_padding_index = random.randint(0, padding_height)
    height_padding = (height_padding_index, padding_height - height_padding_index)
    width_padding = (width_padding_index, padding_width - width_padding_index)

    image_array = np.pad(image_array, (height_padding, width_padding), mode="constant", constant_values=(1.0, 1.0))

    return image_array.astype('float32')


def load_image_names():
    images = []
    directory = ROOT_DIR + "/images"
    add_all_pngs(images, directory)
    return images


def add_all_pngs(images, directory):
    files = os.listdir(directory)
    for file in files:
        filepath = os.path.join(directory, file)
        if os.path.isdir(filepath):
            add_all_pngs(images, filepath)
        elif os.path.isfile(filepath) and filepath.endswith(".png"):
            images.append(filepath)


def load_descriptions():
    descriptions = {}
    directory = ROOT_DIR + "/descriptions"

    with open(os.path.join(directory, "lines.txt")) as file:
        extract_fields(descriptions, file, extract_lines_fields)

    with open(os.path.join(directory, "sentences.txt")) as file:
        extract_fields(descriptions, file, extract_sentences_fields)

    with open(os.path.join(directory, "words.txt")) as file:
        extract_fields(descriptions, file, extract_words_fields)

    return descriptions


def extract_fields(descriptions, file, extract_method):
    while True:
        line = file.readline()
        if not line:
            break
        if not line.startswith("#"):
            extract_method(descriptions, line)


def extract_lines_fields(descriptions, line):
    fields = line.split(maxsplit=8)
    label = fields[-1][:-1].replace(" ", "|")
    descriptions[fields[0]] = [fields[1], fields[-3], fields[-2], label]


def extract_words_fields(descriptions, line):
    fields = line.split(maxsplit=8)
    label = fields[-1][:-1].replace(" ", "|")
    descriptions[fields[0]] = [fields[1], fields[-4], fields[-3], label]


def extract_sentences_fields(descriptions, line):
    fields = line.split(maxsplit=9)
    label = fields[-1][:-1].replace(" ", "|")
    descriptions[fields[0]] = [fields[2], fields[-3], fields[-2], label]


if __name__ == '__main__':
    main()
