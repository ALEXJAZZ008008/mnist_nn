from __future__ import division, print_function
import os
import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import idx2numpy

import network


def get_x_train(input_path, file_name):
    print("Getting x train")

    x_train = idx2numpy.convert_from_file(input_path + file_name)
    x_train = np.expand_dims(x_train, axis=3)

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
              epochs,
              lr,
              lr_factor):
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

            output_size = 10

            input_x = k.layers.Input(x_train.shape[1:])

            x = network.simple_resnet(input_x, output_size)

            x = network.output_module(x, output_size)

            model = k.Model(inputs=input_x, outputs=x)

            model.compile(optimizer=k.optimizers.SGD(lr=lr),
                          loss=k.losses.sparse_categorical_crossentropy,
                          metrics=["accuracy"])

    else:
        print("Using input model")

        model = input_model

    tf.compat.v1.keras.backend.set_value(model.optimizer.lr, lr)

    print("lr: " + str(k.backend.eval(model.optimizer.lr)))

    model.summary()
    k.utils.plot_model(model, output_path + "model.png")

    print("Fitting model")

    tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.local_variables_initializer())

    y_train_len = len(y_train)
    batch_size = int(y_train_len / 4)

    if batch_size <= 0:
        batch_size = 1

    patience = int(epochs / 11)

    if patience <= 0:
        patience = 1

    reduce_lr = k.callbacks.ReduceLROnPlateau(monitor="acc",
                                              factor=lr_factor,
                                              patience=patience,
                                              cooldown=1,
                                              verbose=0)

    loss = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)

    print("Metrics: ", model.metrics_names)
    print("Train loss, acc:", loss)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr], verbose=1)

    loss = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)

    print("Metrics: ", model.metrics_names)
    print("Train loss, acc:", loss)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if save_bool:
        model.save(output_path + "/model.h5")

    if apply_bool:
        test_model(model, input_path, test_data_name, test_label_name, input_path, output_path)

    return model, k.backend.eval(model.optimizer.lr)


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

    output = np.reshape(np.asfarray(output), (-1, 1))

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
        write_to_file(file, np.reshape(np.asfarray(difference), (-1, 1)))

    return output


def main(fit_model_bool, while_bool, mnist_bool):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    tf.compat.v1.keras.backend.set_session(sess)

    tf.compat.v1.keras.backend.set_floatx("float16")

    if fit_model_bool:
        while_model = None

        while True:
            print("Fit model")

            lr = 1.0
            lr_factor = 0.9

            if mnist_bool:
                while_model, lr = fit_model(while_model,
                                            True,
                                            while_bool,
                                            True,
                                            "./data/mnist/",
                                            "/train-images-idx3-ubyte",
                                            "/train-labels-idx1-ubyte",
                                            "/t10k-images-idx3-ubyte",
                                            "/t10k-labels-idx1-ubyte",
                                            "./results/",
                                            1000,
                                            lr,
                                            lr_factor)
            else:
                while_model, lr = fit_model(while_model,
                                            True,
                                            while_bool,
                                            True,
                                            "./data/fashion/",
                                            "/train-images-idx3-ubyte",
                                            "/train-labels-idx1-ubyte",
                                            "/t10k-images-idx3-ubyte",
                                            "/t10k-labels-idx1-ubyte",
                                            "./results/",
                                            1000,
                                            lr,
                                            lr_factor)

            if not while_bool:
                break
    else:
        print("Test model")

        if mnist_bool:
            test_model(None,
                       "./data/mnist/",
                       "/t10k-images-idx3-ubyte",
                       "/t10k-labels-idx1-ubyte",
                       "./results/",
                       "./results/")
        else:
            test_model(None,
                       "./data/fashion/",
                       "/t10k-images-idx3-ubyte",
                       "/t10k-labels-idx1-ubyte",
                       "./results/",
                       "./results/")


if __name__ == "__main__":
    main(True, False, True)
