from test_net import test_mnist, test_human, simple_test_cube, simple_test_hexagon

#mnist_path = r'.\MNIST'
simple_data_path = '../SN_projekt2/klastrowanie'
mnist_path = '../mnist'
human_path = '../UCI_HAR_Dataset'

#simple_test_cube(simple_data_path, 0.9)
simple_test_hexagon(simple_data_path, 0.9)
#test_mnist(mnist_path)
#test_human(human_path)

