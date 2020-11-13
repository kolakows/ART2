from mnist import MNIST
from net import Art2
import matplotlib.pyplot as plt

mndata = MNIST(r'.\MNIST')
img, labels = mndata.load_training()

L1_size = 784
L2_size = 20

art2 = Art2(L1_size, L2_size, d = 0.9, c = 0.1, rho = 0.9)
art2.train(img[:10000], epochs = 1)

clusters = []
plt.figure()
f, axarr = plt.subplots(L2_size//5,5)
axarr = axarr.flatten()
for i in range(L2_size):
    cluster = art2.get_cluster_exemplar(i)
    clusters.append(cluster)
    axarr[i].imshow(cluster.reshape(28,28))
plt.show()

