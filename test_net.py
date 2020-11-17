from mnist import MNIST
from net import Art2
from art2 import Art2Network 
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def simple_test_cube(directory_path, train_split = 0.9):
    data = pd.read_csv(f'{directory_path}/cube.csv').to_numpy()
    np.random.shuffle(data)
    test_count = int(train_split * len(data))

    x_train = data[:test_count, :-1]
    y_train = data[:test_count, -1]
    x_test = data[test_count:, :-1]
    y_test = data[test_count:, -1]
    
    L1_size = 3
    L2_size = 8

    # art2 = Art2(L1_size, L2_size, d = 0.9, c = 0.1, rho = 0.9)
    # art2.train(img[:10000], epochs = 1)

    net = Art2Network(L1_size, L2_size, 0.9)
    y_train_pred = net.process_points(x_train, True)
    y_test_pred = net.process_points(x_test, False)

    print(f'Train accuracy: {cluster_acc(y_train, y_train_pred)}')
    print(f'Test accuracy: {cluster_acc(y_test, y_test_pred)}')
    show_confusion_matrix(y_test, y_test_pred)

    Plot3dData(x_train, y_train_pred, net, y_train, 'Simple cube')

def simple_test_cube_not_matching(directory_path):
    data = pd.read_csv(f'{directory_path}/cube.csv').to_numpy()
    data_not_matching = pd.read_csv(f'{directory_path}/cube-notmatching.csv').to_numpy()

    x_train = data[:, :-1]
    y_train = data[:, -1]
    x_test = data_not_matching[:, :-1]
    y_test = data_not_matching[:, -1]
    
    L1_size = 3
    L2_size = 8

    # art2 = Art2(L1_size, L2_size, d = 0.9, c = 0.1, rho = 0.9)
    # art2.train(img[:10000], epochs = 1)

    net = Art2Network(L1_size, L2_size, 0.9)
    y_train_pred = net.process_points(x_train, True)

    y_test_pred = net.process_points(x_test, False)
    print(f'Train accuracy: {cluster_acc(y_train, y_train_pred)}')
    print(f'Test accuracy: {cluster_acc(y_test, y_test_pred)}')
    show_confusion_matrix(y_test, y_test_pred)

    Plot3dData(x_test, y_test_pred, net, y_test, 'cube-notmaching')

def simple_test_hexagon(directory_path, train_split = 0.9):
    data = pd.read_csv(f'{directory_path}/hexagon.csv').to_numpy()
    np.random.shuffle(data)
    test_count = int(train_split * len(data))

    x_train = data[:test_count, :-1]
    y_train = data[:test_count, -1]
    x_test = data[test_count:, :-1]
    y_test = data[test_count:, -1]
    
    L1_size = 2
    L2_size = 6

    # art2 = Art2(L1_size, L2_size, d = 0.9, c = 0.1, rho = 0.9)
    # art2.train(img[:10000], epochs = 1)

    net = Art2Network(L1_size, L2_size, 0.99)
    y_train_pred = net.process_points(x_train, True)

    y_test_pred = net.process_points(x_test, False)
    print(f'Train accuracy: {cluster_acc(y_train, y_train_pred)}')
    print(f'Test accuracy: {cluster_acc(y_test, y_test_pred)}')
    show_confusion_matrix(y_test, y_test_pred)

    plot_hex(x_train, y_train_pred, net, y_train)

def test_mnist(directory_path, train_split = 0.9):
    mndata = MNIST(directory_path)
    img, labels_true = mndata.load_training()

    test_count = int(train_split * len(img))

    x_train = img[:test_count]
    y_train = labels_true[:test_count]
    x_test = img[test_count:]
    y_test = labels_true[test_count:]

    L1_size = 784
    L2_size = 10

    # art2 = Art2(L1_size, L2_size, d = 0.9, c = 0.1, rho = 0.9)
    # art2.train(img[:10000], epochs = 1)

    net = Art2Network(L1_size, L2_size, 0.8)
    
    y_train_pred = net.process_points(x_train, True)
    y_test_pred = net.process_points(x_test, False)

    print(f'Train accuracy: {cluster_acc(y_train, y_train_pred)}')
    print(f'Test accuracy: {cluster_acc(y_test, y_test_pred)}')
    show_confusion_matrix(y_test, y_test_pred)

    clusters = []
    f, axarr = plt.subplots(L2_size//5,5)
    axarr = axarr.flatten()
    for i in range(L2_size):
        #cluster = art2.get_cluster_exemplar(i)
        cluster = net.get_cluster_exemplar(i)
        clusters.append(cluster)
        axarr[i].imshow(cluster.reshape(28,28))
    plt.show()

def test_mnist_subset(directory_path, train_split = 0.9, example_count = 'all'):
    mndata = MNIST(directory_path)
    img, labels_true = mndata.load_training()
    if example_count != 'all':
        img = img[:example_count]
        labels_true = labels_true[:example_count]

    test_count = int(train_split * len(img))

    x_train = np.array(img[:test_count])
    y_train = np.array(labels_true[:test_count])
    
    #data for only 8 labels
    x_train = x_train[y_train < 8]
    y_train = y_train[y_train < 8]

    #test data with all labels
    x_test = img[test_count:]
    y_test = labels_true[test_count:]

    L1_size = 784
    L2_size = 10

    # art2 = Art2(L1_size, L2_size, d = 0.9, c = 0.1, rho = 0.9)
    # art2.train(img[:10000], epochs = 1)

    net = Art2Network(L1_size, L2_size, 0.8)
    
    y_train_pred = net.process_points(x_train, True)
    y_test_pred = net.process_points(x_test, False)

    show_cms(y_train, y_train_pred, y_test, y_test_pred)

    print(f'Train accuracy: {cluster_acc(y_train, y_train_pred)}')
    print(f'Test accuracy: {cluster_acc(y_test, y_test_pred)}')
    # show_confusion_matrix(y_train, y_train_pred)
    # show_confusion_matrix(y_test, y_test_pred)

    clusters = []
    f, axarr = plt.subplots(L2_size//5,5)
    axarr = axarr.flatten()
    for i in range(L2_size):
        #cluster = art2.get_cluster_exemplar(i)
        cluster = net.get_cluster_exemplar(i)
        clusters.append(cluster)
        axarr[i].imshow(cluster.reshape(28,28))
    plt.show()

def test_human(directory_path):
    x_train = pd.read_csv(f'{directory_path}/train/X_train.txt', delim_whitespace=True, header=None).to_numpy()
    y_train = pd.read_csv(f'{directory_path}/train/y_train.txt', delim_whitespace=True, header=None).to_numpy()
    x_test = pd.read_csv(f'{directory_path}/test/X_test.txt', delim_whitespace=True, header=None).to_numpy()
    y_test = pd.read_csv(f'{directory_path}/test/y_test.txt', delim_whitespace=True, header=None).to_numpy()
    
    L1_size = 561
    L2_size = 6

    # art2 = Art2(L1_size, L2_size, d = 0.9, c = 0.1, rho = 0.9)
    # art2.train(img[:10000], epochs = 1)

    net = Art2Network(L1_size, L2_size, 0.95)
    y_train_pred = net.process_points(x_train, True)

    y_test_pred = net.process_points(x_test, False)
    print(f'Test accuracy: {cluster_acc(y_train, y_train_pred)}')
    print(f'Train accuracy: {cluster_acc(y_test, y_test_pred)}')
    show_confusion_matrix(y_test, y_test_pred)