import os
import random
import string
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_WIDTH = 1024
TARGET_HEIGHT = 32
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

MAX_STRING_LENGTH = 256
ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase + "!?.,;:-'\"%$&/()=*#+|"
NUM_CLASSES = len(ALPHABET) + 1  # + 1 for CTC blank-symbol (separator between characters)

BATCH_SIZE = 32
EPOCHS = 2
SPLIT_FACTOR = 0.001


def main():
    # filename: [ok, width, height, label]
    descriptions = load_descriptions()
    images = load_image_names()
    np.random.shuffle(images)
    split_index = int(len(images) * SPLIT_FACTOR)

    train(images, split_index, descriptions)
    # test(images, split_index, descriptions, model)


def test(images, split_index, descriptions, model):
    test_images = images[split_index:]
    test_labels = [descriptions[os.path.splitext(os.path.basename(image))[0]][3] for image in test_images]
    test_labels = np.array([np.array([ALPHABET.index(c) for _, c in enumerate(label)]) for label in test_labels])
    test_images = np.array([load_image(image) for image in test_images])


def train(images, split_index, descriptions):
    #print(len(images))
    training_images = images[:split_index]
    training_labels = [descriptions[os.path.splitext(os.path.basename(image))[0]][3] for image in training_images]
    training_labels = np.asarray([text_to_label(label) for label in training_labels])
    training_labels_lengths = np.asarray([np.asarray([len(label)]).astype('int64') for label in training_labels])

    training_images = np.asarray([load_image(image) for image in training_images])

    training_pred_lengths = np.asarray([np.asarray([MAX_STRING_LENGTH]).astype('int64') for _ in training_labels])
    print(training_images[0])
    print(training_labels[0])
    print(training_pred_lengths[0])
    print(training_labels_lengths[0])

    model = train_model(training_images, training_labels, training_pred_lengths, training_labels_lengths)
    return model


def text_to_label(text):
    label = np.ones([MAX_STRING_LENGTH])
    label *= len(ALPHABET)
    for i, c in enumerate(text):
        label[i] = ALPHABET.index(c)

    return np.asarray(label).astype('int64')


def train_model(training_images, training_labels, training_pred_lengths, training_labels_lengths):
    kernel = (3, 3)

    input_data = layers.Input(name='the_input', shape=(TARGET_WIDTH, TARGET_HEIGHT, 1), dtype='float32')
    # shape(batchsize, MAX_STRING_LENGTH)
    labels = layers.Input(name='the_labels', shape=[MAX_STRING_LENGTH], dtype='int64')
    # shape(batchsize, 1) eg. [[timesteps], [timesteps]]
    prediction_length = layers.Input(name='prediction_length', shape=[1], dtype='int64')
    # shape(batchsize, 1) eg. [[3], [4]] for words "aaa" and "aaaa"
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    inner = layers.Conv2D(name='conv1', filters=16, kernel_size=kernel, activation=tf.nn.relu, padding='same')(
        input_data)
    inner = layers.MaxPooling2D(name='max_pool1', pool_size=(2, 2))(inner)
    inner = layers.Conv2D(name='conv2', filters=16, kernel_size=kernel, activation=tf.nn.relu, padding='same')(inner)
    inner = layers.MaxPooling2D(name='max_pool2', pool_size=(2, 2))(inner)
    inner = layers.Reshape(name='reshape1', target_shape=(256, 128))(inner)
    inner = layers.Dense(256, name='dense1', activation=tf.nn.relu)(inner)
    inner = layers.Bidirectional(layers.GRU(512, return_sequences=True), name='bidir1')(inner)
    inner = layers.Bidirectional(layers.GRU(512, return_sequences=True), name='bidir2')(inner)
    inner = layers.Dense(NUM_CLASSES, name='dense2')(inner)
    y_pred = layers.Activation('softmax', name='softmax')(inner)  # shape(batchsize, timesteps, NUM_CLASSES)

    loss_out = layers.Lambda(ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, prediction_length, label_length])

    sgd = keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = keras.models.Model(inputs=[input_data, labels, prediction_length, label_length], outputs=loss_out)

    print("inputs")
    [print(i.shape, i.dtype) for i in model.inputs]
    print("outputs")
    [print(o.shape, o.dtype) for o in model.outputs]
    print("layers")
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    model.summary()

    # Train the digit classification model
    model.fit([training_images, training_labels, training_pred_lengths, training_labels_lengths],
              np.zeros([len(training_images)]), batch_size=BATCH_SIZE, epochs=EPOCHS)
    return model


def ctc_decode(prediction, input_length):
    return K.get_value(K.ctc_decode(prediction, input_length))


def ctc_loss(args):
    prediction, labels, prediction_length, label_length = args
    loss = K.ctc_batch_cost(labels, prediction, prediction_length, label_length)
    print(loss)
    return loss


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
    image_array = image_array.T

    #print(sys.getsizeof(image_array.astype('float32')))

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
        if os.path.isdir(filepath) and "sentences" not in filepath:
            add_all_pngs(images, filepath)
        elif os.path.isfile(filepath) and filepath.endswith(".png"):
            images.append(filepath)


def load_descriptions():
    descriptions = {}
    directory = ROOT_DIR + "/descriptions"

    with open(os.path.join(directory, "lines.txt")) as file:
        extract_fields(descriptions, file, extract_lines_fields)

    # with open(os.path.join(directory, "sentences.txt")) as file:
    # extract_fields(descriptions, file, extract_sentences_fields)

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
