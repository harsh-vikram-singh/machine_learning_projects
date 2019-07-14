from utils import *

# load MNIST data
train_x, train_y, test_x, test_y = get_MNIST_data()

# plot first 20 images of the training set
plot_images(test_x[0:20, :])
