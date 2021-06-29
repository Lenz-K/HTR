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
SEPARATOR_SYMBOL = len(ALPHABET)

BATCH_SIZE = 64
EPOCHS = 5
SPLIT_FACTOR = 0.1


def main():
    # filename: [ok, width, height, label]
    descriptions = load_descriptions()
    images = load_image_names()
    np.random.shuffle(images)
    split_index = int(len(images) * SPLIT_FACTOR)

    train(images, split_index, descriptions)


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


def train(images, split_index, descriptions):
    print(f"Number of loaded images: {len(images)}")
    training_images = images[:split_index]
    print(f"Training set size: {len(training_images)}")
    training_labels = [descriptions[os.path.splitext(os.path.basename(image))[0]][3] for image in training_images]

    for i in range(len(training_images)):
        training_images[i], training_labels[i] = load_images_and_create_labels(training_images[i], training_labels[i])

    training_labels_lengths = np.asarray([np.asarray([len(label)]).astype('int64') for label in training_labels])
    training_pred_lengths = np.asarray([np.asarray([MAX_STRING_LENGTH]).astype('int64') for _ in training_labels])

    print(len(training_images))
    print(training_images[0])
    print(len(training_labels))
    print(training_labels[0])
    print(len(training_pred_lengths))
    print(training_pred_lengths[0])
    print(len(training_labels_lengths))
    print(training_labels_lengths[0])

    model = train_model(np.asarray(training_images), np.asarray(training_labels), training_pred_lengths, training_labels_lengths)
    return model


def text_to_label(text, word_width, width_padding_index):
    begin = width_padding_index // 4
    word_width = word_width // 4
    step_width = max(word_width // len(text), 1)

    label = np.ones([MAX_STRING_LENGTH])
    label *= SEPARATOR_SYMBOL

    print(f"Begin: {begin} available steps: {word_width} number of chars: {len(text)} step width: {step_width}")

    j = begin
    for i, c in enumerate(text):
        space_for_c = (step_width + 1) // 2
        space_for_separator = step_width // 2
        for _ in range(space_for_c):
            label[j] = ALPHABET.index(c)
            j += 1
        j += space_for_separator

    return np.asarray(label).astype('int64')


def bad_text_to_label(text):
    label = np.ones([MAX_STRING_LENGTH])
    label *= SEPARATOR_SYMBOL

    for i, c in enumerate(text):
        label[i] = ALPHABET.index(c)

    return np.asarray(label).astype('int64')


def load_images_and_create_labels(image_path, label):
    image = Image.open(image_path)
    image.thumbnail(TARGET_SIZE)
    image_array = np.asarray(image)
    #print(image_array.shape)
    image_array = image_array / 255.0
    image_width = len(image_array[0])
    image_height = len(image_array)

    padding_width = TARGET_WIDTH - image_width
    padding_height = TARGET_HEIGHT - image_height
    width_padding_index = random.randint(0, padding_width)
    height_padding_index = random.randint(0, padding_height)
    height_padding = (height_padding_index, padding_height - height_padding_index)
    width_padding = (width_padding_index, padding_width - width_padding_index)

    image_array = np.pad(image_array, (height_padding, width_padding), mode="constant", constant_values=(1.0, 1.0))
    image_array = image_array.T
    #print(image_array.shape)

    # print(sys.getsizeof(image_array.astype('float32')))

    return image_array.astype('float32'), bad_text_to_label(label)  # , image_width, width_padding_index)


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


if __name__ == '__main__':
    main()
