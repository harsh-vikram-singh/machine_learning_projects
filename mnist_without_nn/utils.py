import pickle
import gzip
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def read_pickle_data(file_name):
    '''reads a pickle file and returns the data

    Input ->

    file_name: a compressed(gzip) pickle file

    Returns ->

    file data, that is formatted in the following
    format: training_set, validation_set and testing_set
    '''
    f = gzip.open(file_name, 'rb')
    data = pickle.load(f, encoding='latin1')
    f.close()
    return data


def get_MNIST_data():
    '''gets all the files in folder called Dataset, 
    combines the training and validation set, then returns
    the training and testing set

    Input -> None

    Returns ->

    a tuple in the following format:
    (training_inputs, training_labels, test_inputs, test_labels)
    '''
    train_set, valid_set, test_set = read_pickle_data('Datasets/mnist.pkl.gz')
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    train_x = np.vstack((train_x, valid_x))
    train_y = np.append(train_y, valid_y)
    test_x, test_y = test_set
    return (train_x, train_y, test_x, test_y)


def plot_images(X):
    if X.ndim == 1:
        X = np.array([X])
    num_images = X.shape[0]
    num_rows = math.floor(math.sqrt(num_images))
    num_cols = math.ceil(num_images/num_rows)
    for i in range(num_images):
        reshaped_image = X[i, :].reshape(28, 28)
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(reshaped_image, cmap=cm.Greys_r)
        plt.axis('off')
    plt.show()
