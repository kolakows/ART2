import numpy as np
from numpy import linalg as LA

class Art2():
    def __init__(self,L1_size, L2_size, d, c, rho):
        self.Wbu = np.full([L1_size, L2_size] , 1/((1-d) * np.sqrt(L1_size)))
        self.Wtd = np.zeros([L2_size, L1_size])
        self.L1 = np.zeros(L1_size)
        self.L2 = np.zeros(L2_size)
        self.d = d
        self.c = c
        self.rho = rho

    def update_weights(self, x, j):
        # dont get the (self.d - 1) part
        self.Wbu[:,j] += self.d * (x + (self.d - 1) * self.Wbu[:,j]) 
        self.Wtd[j,:] += self.d * (x + (self.d - 1) * self.Wtd[j,:])

        # normalise?
        # self.Wbu[:,j] = self.normalise(self.Wbu[:,j])
        # self.Wtd[j,:] = self.normalise(self.Wtd[j,:])

    def update_L1(self, x):
        self.L1 = x

    def calculate_L2(self):
        self.L2 = np.dot(self.L1, self.Wbu)
    
    def get_cluster_exemplar(self, j):
        # instead of multipying by e_j vector (e_j * Wtd), we can implicitly extract j-th row
        return self.Wtd[j,:]
        
    def normalise(self, x):
        return x/LA.norm(x)

    def train(self, data, epochs):
        for i in range(epochs):
            for x in data:
                self.train_sample(x)

    def train_sample(self, x):
        l2 = self.forward(x)
        r_norms = []
        for i in range(l2.shape[0]):
            # find current max
            j = np.argmax(l2)
            l2[j] = -np.inf
            # get exemplar from top-down weights
            y = self.get_cluster_exemplar(j)
            # compare and update
            r_norm = self.reset_norm(x,y)
            if r_norm >= self.rho:
                self.update_weights(x, j)
                break
            r_norms.append(r_norm)
        if np.all((l2 == -np.inf)):
            # update at best possible match
            j = np.argmax(r_norms)
            self.update_weights(x, j)
        self.reset_layers()

    def reset_norm(self, x, y):
        # norm(x + y) <= norm(x) + norm(y), so r_norm <= 1, 
        r_norm = LA.norm((x + self.c * y))/(LA.norm(x) + LA.norm(self.c * y)) 
        return r_norm

    def forward(self, x):
        x = self.normalise(x)
        self.update_L1(x)
        self.calculate_L2()
        return self.L2.copy()  

    def reset_layers(self):
        self.L1.fill(0)
        self.L2.fill(0)

 