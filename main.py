from __future__ import division, print_function
import os
from tensorflow import keras as k
import numpy as np
import idx2numpy


def get_x_train(input_path, file_name):
    print("Getting x train")

    x_train = idx2numpy.convert_from_file(input_path + file_name)

    print("Got x train")

    return np.nan_to_num(x_train).astype(np.float)


def get_y_train(input_path, file_name):
    print("Get y train")

    y_train = idx2numpy.convert_from_file(input_path + file_name)

    print("Got y train")

    return np.nan_to_num(y_train).astype(np.float)


def write_to_file(file, data):
    for i in range(len(data)):
        output_string = ""

        for j in range(len(data[i])):
            output_string = output_string + str(data[i][j]) + ','

        output_string = output_string[:-1] + '\n'

        file.write(output_string)


def fit_model(input_model, load_bool, apply_bool, input_path, data_name, label_name, output_path, epochs):
    print("Get training data")

    x_train = get_x_train(input_path, data_name)
    y_train = get_y_train(input_path, label_name)

    if input_model is None:
        print("No input model")

        if load_bool:
            print("Load model from file")

            model = k.models.load_model(output_path + "/model.h5")
        else:
            print("Generate new model")

            input_x = k.layers.Input(x_train.shape[1:])

            x = k.layers.UpSampling1D(10)(input_x)  # upsample by factor of 2

            # 5 x 5 x (previous num channels = 1) kernels (32 times => 32 output channels)
            # padding='same' for zero padding
            x = k.layers.Conv1D(32, 5, activation=k.activations.relu, padding='same')(x)
            x = k.layers.AveragePooling1D(2)(x)  # downsample by factor of 2, using "average" interpolation

            # 3 x 3 x (previous num channels = 32) kernels (64 times)
            x = k.layers.Conv1D(64, 3, activation=k.activations.relu, padding='same')(x)
            x = k.layers.AveragePooling1D(2)(x)  # downsample by factor of 2, using "average" interpolation

            # 3 x 3 x (previous num channels = 64) kernels (128 times)
            x = k.layers.Conv1D(128, 3, activation=k.activations.relu, padding='same')(x)
            x = k.layers.AveragePooling1D(2)(x)  # downsample by factor of 2, using "average" interpolation

            # 3 x 3 x (previous num channels = 128) kernels (256 times)
            x = k.layers.Conv1D(256, 3, activation=k.activations.relu, padding='same')(x)
            x = k.layers.AveragePooling1D(2)(x)  # downsample by factor of 2, using "average" interpolation

            # 1 x 1 x (previous num channels = 256) kernels (256 times)
            x = k.layers.Conv1D(256, 1, activation=k.activations.relu, padding='same')(x)

            # 1 x 1 x (previous num channels = 256) kernels (256 times)
            x = k.layers.Conv1D(256, 1, activation=k.activations.relu, padding='same')(x)

            # 1 x 1 x (previous num channels = 256) kernels (256 times)
            x = k.layers.Conv1D(256, 1, activation=k.activations.relu, padding='same')(x)

            x = k.layers.Flatten()(x)  # vectorise

            x = k.layers.Dense(256, activation=k.activations.relu)(x)  # traditional neural layer with 256 outputs
            x = k.layers.Dropout(0.20)(x)  # discard 20% outputs

            x = k.layers.Dense(1280, activation=k.activations.relu)(x)  # traditional neural layer with 1280 outputs
            x = k.layers.Dropout(0.50)(x)  # discard 50% outputs

            x = k.layers.Dense(10, activation=k.activations.softmax)(x)  # 10 outputs

            model = k.Model(input_x, x)

            # losses:  K.losses.*
            # optimisers: K.optimizers.*
            model.compile(optimizer=k.optimizers.Nadam(), loss=k.losses.sparse_categorical_crossentropy)
    else:
        print("Using input model")

        model = input_model

    model.summary()

    print("Fitting model")

    model.fit(x_train, y_train, epochs=epochs, verbose=1)

    loss = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', loss)

    print("Saving model")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model.save(output_path + "/model.h5")

    if apply_bool:
        test_model(model, input_path, data_name, label_name, input_path, output_path)

    return model


def test_model(input_model, data_input_path, data_input_name, data_input_label_name, model_input_path, output_path):
    print("Get test data")

    x_test = get_x_train(data_input_path, data_input_name)
    y_test = get_y_train(data_input_path, data_input_label_name)

    if input_model is None:
        print("No input model")
        print("Load model from file")

        model = k.models.load_model(model_input_path + "/model.h5")
    else:
        model = input_model

    print("Applying model")

    model_output = model.predict(x_test)
    output = []

    for i in range(len(model_output)):
        likelihood = -1.0
        index = -1

        for j in range(len(model_output[i])):
            if model_output[i][j] > likelihood:
                likelihood = model_output[i][j]
                index = j

        output.append(np.array(index))

    output = np.reshape(np.asarray(output), (-1, 1))

    with open(output_path + "/output.csv", 'w') as file:
        write_to_file(file, output)

    output = output.squeeze()
    difference = []

    for i in range(len(output)):
        if output[i] == y_test[i]:
            difference.append(np.array(0))
        else:
            difference.append(np.array(1))

    print("Difference: " + str(sum(difference)))

    with open(output_path + "/difference.csv", 'w') as file:
        write_to_file(file, np.reshape(np.asarray(difference), (-1, 1)))


if __name__ == "__main__":
    fit_model_bool = True
    while_bool = True

    if fit_model_bool:
        while_model = None

        while True:
            print("Fit model")

            fit_model(while_model,
                      while_bool,
                      True,
                      "./data/",
                      "/train-images-idx3-ubyte",
                      "/train-labels-idx1-ubyte",
                      "./results/",
                      10)

            if not while_bool:
                break
    else:
        print("Test model")
        
        test_model(None,
                   "./data/",
                   "/t10k-images-idx3-ubyte",
                   "/t10k-labels-idx1-ubyte",
                   "./results/",
                   "./results/")