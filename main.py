from __future__ import division, print_function
import os
from tensorflow import keras as k
import numpy as np
import idx2numpy

import network
import network_regularisation


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


def fit_model(input_model,
              save_bool,
              load_bool,
              apply_bool,
              input_path,
              data_name,
              label_name,
              test_data_name,
              test_label_name,
              output_path,
              epochs):
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

            x = network_regularisation.conv_fully_connected(input_x)

            x = network.output_module(x)

            model = k.Model(input_x, x)

            model.compile(optimizer=k.optimizers.Nadam(),
                          loss=k.losses.sparse_categorical_crossentropy,
                          metrics=["accuracy"])
    else:
        print("Using input model")

        model = input_model

    model.summary()
    k.utils.plot_model(model, output_path + "model.png")

    print("Fitting model")

    model.fit(x_train, y_train, epochs=epochs, verbose=1)

    loss = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', loss)

    print("Saving model")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if save_bool:
        model.save(output_path + "/model.h5")

    if apply_bool:
        test_model(model, input_path, test_data_name, test_label_name, input_path, output_path)

    return model


def write_to_file(file, data):
    for i in range(len(data)):
        output_string = ""

        for j in range(len(data[i])):
            output_string = output_string + str(data[i][j]) + ','

        output_string = output_string[:-1] + '\n'

        file.write(output_string)


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

    print("Difference: " + str((sum(difference) / len(y_test)) * 100) + "%")

    with open(output_path + "/difference.csv", 'w') as file:
        write_to_file(file, np.reshape(np.asarray(difference), (-1, 1)))


def main(fit_model_bool, while_bool, mnist_bool):
    if fit_model_bool:
        while_model = None

        while True:
            print("Fit model")

            if mnist_bool:
                while_model = fit_model(while_model,
                                        True,
                                        while_bool,
                                        True,
                                        "./data/mnist/",
                                        "/train-images-idx3-ubyte",
                                        "/train-labels-idx1-ubyte",
                                        "/t10k-images-idx3-ubyte",
                                        "/t10k-labels-idx1-ubyte",
                                        "./results/",
                                        10)
            else:
                while_model = fit_model(while_model,
                                        True,
                                        while_bool,
                                        True,
                                        "./data/fashion/",
                                        "/train-images-idx3-ubyte",
                                        "/train-labels-idx1-ubyte",
                                        "/t10k-images-idx3-ubyte",
                                        "/t10k-labels-idx1-ubyte",
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


if __name__ == "__main__":
    main(True, False, False)
