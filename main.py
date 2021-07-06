import os
import random
import string

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras import layers

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_WIDTH = 512
TARGET_HEIGHT = 32
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

MAX_STRING_LENGTH = 62
ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase + "!?.,;:-'\"&/()*#+|"
NUM_CLASSES = len(ALPHABET) + 1  # + 1 for blank-symbol
SEPARATOR_SYMBOL = len(ALPHABET)

BATCH_SIZE = 64
EPOCHS = 25  # TODO
SPLIT_FACTOR = 0.99  # TODO


def main():
    # filename: [ok, width, height, label]
    descriptions = load_descriptions()
    images = load_image_names()
    print(f"Number of loaded images: {len(images)}")
    np.random.shuffle(images)
    split_index = int(len(images) * SPLIT_FACTOR)

    model = train_model(images, split_index, descriptions)

    #test_images, test_labels = get_datasets(images[len(images) - 100:], descriptions)  # TODO
    test_images, test_labels = get_datasets(images[split_index:], descriptions)  # TODO
    print(f"Test set size: {len(test_images)}")
    model.evaluate(test_images, test_labels)

    for i in range(10):
        image = test_images[i]
        pyplot.figure(figsize=(5, 5))
        pyplot.imshow(image)
        pyplot.show()
        print()
        print(f"     Truth: {label_to_text(test_labels[i])}")
        predict_image(model, image)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter.convert()

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    with open("htr_float_model.tflite", "wb") as f:
        f.write(tflite_float_model)
    with open("htr_small_model.tflite", "wb") as f:
        f.write(tflite_quantized_model)


def train_model(images, split_index, descriptions):
    training_images, training_labels = get_datasets(images[:split_index], descriptions)
    print(f"Training set size: {len(training_images)}")
    print(training_images[0])
    print(training_images[0].shape)
    print(training_labels[0])

    pyplot.figure(figsize=(5, 5))
    pyplot.imshow(training_images[0])
    pyplot.show()

    model = create_model()

    # Train the digit classification model
    model.fit(training_images, training_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

    training_images, training_labels = (0, 0)
    print(training_images)
    print(training_labels)

    return model


def get_datasets(images, descriptions):
    labels = [descriptions[os.path.splitext(os.path.basename(image))[0]][3] for image in images]

    for i in range(len(images)):
        images[i] = load_image(images[i])
        labels[i] = text_to_label(labels[i])

    return np.asarray(images), np.asarray(labels)


def create_model():
    kernel = (3, 3)

    input_data = layers.Input(name='the_input', shape=(TARGET_HEIGHT, TARGET_WIDTH), dtype='float32')

    inner = layers.Reshape(name='reshape0', target_shape=(TARGET_HEIGHT, TARGET_WIDTH, 1))(input_data)
    inner = layers.Conv2D(name='conv1', filters=16, kernel_size=kernel, activation=tf.nn.relu)(inner)
    inner = layers.MaxPooling2D(name='max_pool1', pool_size=(2, 2))(inner)
    inner = layers.Conv2D(name='conv2', filters=16, kernel_size=kernel, activation=tf.nn.relu)(inner)
    inner = layers.MaxPooling2D(name='max_pool2', pool_size=(2, 2))(inner)
    inner = layers.Conv2D(name='conv3', filters=16, kernel_size=kernel, activation=tf.nn.relu)(inner)
    inner = layers.MaxPooling2D(name='max_pool3', pool_size=(2, 2))(inner)

    inner = layers.Reshape(name='reshape1', target_shape=(MAX_STRING_LENGTH, 32))(inner)

    inner = layers.Dense(512, name='dense1', activation=tf.nn.relu)(inner)
    inner = layers.Dense(512, name='dense2', activation=tf.nn.relu)(inner)
    inner = layers.Dense(512, name='dense41', activation=tf.nn.relu)(inner)

    inner = layers.Bidirectional(layers.LSTM(512, return_sequences=True), name='bidir1')(inner)
    inner = layers.Bidirectional(layers.LSTM(512, return_sequences=True), name='bidir2')(inner)

    inner = layers.Dense(512, name='dense3', activation=tf.nn.relu)(inner)
    inner = layers.Dense(512, name='dense4', activation=tf.nn.relu)(inner)
    inner = layers.Dense(512, name='dense42', activation=tf.nn.relu)(inner)

    # inner = layers.Dropout(0.25)(inner)
    # inner = layers.Flatten()(inner)

    outputs = layers.Dense(NUM_CLASSES, name='dense5', activation="softmax")(inner)

    model = keras.models.Model(inputs=input_data, outputs=outputs)

    # print("inputs")
    # [print(i.shape, i.dtype) for i in model.inputs]
    # print("outputs")
    # [print(o.shape, o.dtype) for o in model.outputs]
    # print("layers")
    # [print(l.name, l.input_shape, l.dtype) for l in model.layers]

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    return model


def predict_image(model, image):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)

    text = ""
    for c_probabilities in prediction[0]:
        max_p = 0
        max_p_i = -1
        for i in range(NUM_CLASSES):
            if c_probabilities[i] > max_p:
                max_p = c_probabilities[i]
                max_p_i = i

        if max_p_i != SEPARATOR_SYMBOL:
            text += ALPHABET[max_p_i]
        else:
            text += "="
    print(f"Prediction: {text}")
    return text


def text_to_label(text):
    label = np.ones([MAX_STRING_LENGTH])
    label *= SEPARATOR_SYMBOL

    # label = np.ones([len(text)])

    for i, c in enumerate(text[:MAX_STRING_LENGTH]):
        label[i] = ALPHABET.index(c)

    return label.astype('int64')


def label_to_text(label):
    res = ""

    for c in label:
        if c == SEPARATOR_SYMBOL:
            res += "="
        else:
            res += ALPHABET[c]

    return res


def load_image(image_path):
    image = Image.open(image_path)
    image.thumbnail(TARGET_SIZE)
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    image_width = len(image_array[0])
    image_height = len(image_array)

    padding_width = TARGET_WIDTH - image_width
    padding_height = TARGET_HEIGHT - image_height
    width_padding_index = 0
    height_padding_index = random.randint(0, padding_height)
    height_padding = (height_padding_index, padding_height - height_padding_index)
    width_padding = (width_padding_index, padding_width - width_padding_index)

    image_array = np.pad(image_array, (height_padding, width_padding), mode="constant", constant_values=(1.0, 1.0))
    # image_array -= 0.5
    # print(image_array.shape)

    # print(sys.getsizeof(image_array.astype('float32')))

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
        if os.path.isdir(filepath) and "words" not in filepath:
            add_all_pngs(images, filepath)
        elif os.path.isfile(filepath) and filepath.endswith(".png"):
            images.append(filepath)


def load_descriptions():
    descriptions = {}
    directory = ROOT_DIR + "/descriptions"

    with open(os.path.join(directory, "lines.txt")) as file:
        extract_fields(descriptions, file, extract_lines_fields)

    # with open(os.path.join(directory, "words.txt")) as file:
    #   extract_fields(descriptions, file, extract_words_fields)

    with open(os.path.join(directory, "sentences.txt")) as file:
        extract_fields(descriptions, file, extract_sentences_fields)

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
