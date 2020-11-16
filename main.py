from test_net import test_mnist, test_human, simple_test_cube, simple_test_hexagon, test_mnist_subset, simple_test_cube_not_matching

#mnist_path = r'.\MNIST'
simple_data_path = '../SN_projekt2/klastrowanie'
mnist_path = '../mnist'
human_path = '../UCI_HAR_Dataset'

#simple_test_cube(simple_data_path, 0.9) # jest w miarę ok 
#simple_test_cube_not_matching(simple_data_path) # coś dziwnego z wykresem
#simple_test_hexagon(simple_data_path, 0.9) # wykrywa tylko 4 klastry nawet przy vigiliance=0.99999
#test_mnist(mnist_path) # w miare ok
#test_mnist_subset(mnist_path) # czy może w fazie testów zakwalifikować dane do 
test_human(human_path)

