from math import *
from utils import *
import numpy as np

class Art2Network():
    def __init__(self, dataDimensions, maxClustersCount, vigilance):
        self.capacity = maxClustersCount
        self.a = 10
        self.b = 10
        self.c = 0.1
        self.d = 0.9
        self.e = 0
        self.learningLength = 1 # 1 for slow learning
        self.learningRate = 0.6
        self.dataDim = dataDimensions
        self.theta = 1/sqrt(self.dataDim)
        self.f2f1WeightRatio = 0.95
        self.vigilance = vigilance # rho
        self.reset()

    def reset(self):
        self.f1f2 = np.array([[1.0] * self.capacity] * self.dataDim)
        self.f1f2 = self.f1f2 * self.f2f1WeightRatio /((1 - self.d) * sqrt(self.dataDim))
        self.f2f1 = np.array([[0.0] * self.dataDim] * self.capacity)
        self.nbOfClasses = 0
        self.iterationCounter = 0

    def process_points(self, points, learn_mode):
        self.log_description()
        self.iterationCounter = 0
        clusters = []
        for s in points:
            self.iterationCounter += 1
            print("Processing " + str(self.iterationCounter) + "/" + str(len(points)) + " Clusters count: " + str(self.nbOfClasses))
            clusterid = self.process_point(s, learn_mode)
            clusters += [clusterid]
        return clusters

    def process_point(self, S, learn_mode):
        clusterid = self.process(S, learn_mode)
        return clusterid

    def f1_update_U(self):
        divider = self.e + norm(self.f1_V)
        self.f1_U = np.array([vi / divider for vi in self.f1_V])

    def f1_update_W(self, S):
        self.f1_W = S + self.f1_U * self.a

    def f1_update_P(self, activeNeuronIdx):
        if activeNeuronIdx == -1:
            self.f1_P = self.f1_U
        else:       
            self.f1_P = self.f1_U + self.f2f1[activeNeuronIdx,:] * self.d

    def f1_update_X(self):
        divider = self.e + norm(self.f1_W)
        self.f1_X = [wi / divider for wi in self.f1_W]

    def f1_update_Q(self):
        divider = self.e + norm(self.f1_P) 
        self.f1_Q = [pi / divider for pi in self.f1_P]

    def f1_update_V(self):
        self.f1_V = self.noise_suppr(self.f1_X) + [qi * self.b for qi in self.noise_suppr(self.f1_Q)]  

    def noise_suppr(self, X):
        return np.array([(x if x >= self.theta else 0) for x in X])

    def f1_init(self, S):
        self.f1_U = np.array([0] * len(S))
        self.f1_P = np.array([0] * len(S))
        self.f1_Q = np.array([0] * len(S))
        self.f1_update_W(S)
        self.f1_update_X()
        self.f1_update_V()

    def f1_update(self, S, activeNeuronIdx):
        self.f1_update_U()
        self.f1_update_W(S)
        self.f1_update_P(activeNeuronIdx)
        self.f1_update_X()
        self.f1_update_Q()
        self.f1_update_V()   

    def adjust_weights(self, activeNeuronIdx):
        self.f1f2[:,activeNeuronIdx] = self.f1_U * self.learningRate * self.d + self.f1f2[:,activeNeuronIdx] * (1 + self.learningRate * self.d * (self.d - 1))
        self.f2f1[activeNeuronIdx,:] = self.f1_U * self.learningRate * self.d + self.f2f1[activeNeuronIdx, :] * (1 + self.learningRate * self.d * (self.d - 1))

    def learning(self, S, activeNeuronIdx):
        self.f1_update_W(S)
        self.f1_update_X()
        self.f1_update_Q()
        self.f1_update_V()

        for i in range(self.learningLength):
            self.adjust_weights(activeNeuronIdx)
            self.f1_update(S, activeNeuronIdx)

    def calc_vigilance(self, activeNeuronIdx):
        self.f1_update_U()
        self.f1_update_P(activeNeuronIdx)   
        R = (self.f1_U + self.f1_P * self.c) / (self.e + norm(self.f1_U) + self.c * norm(self.f1_P))
        return norm(R)

    def check_cluster(self, activeNeuronIdx):
        result = self.calc_vigilance(activeNeuronIdx) + self.e >= self.vigilance
        return result
            
    def simulate_F1F2(self):
        Y = np.dot(self.f1f2[:,0:self.nbOfClasses+1].T, self.f1_P.T)
        return Y

    def find_class(self, S):
        activeNeuronIdx = -1
        self.f1_init(S)
        self.f1_update(S, activeNeuronIdx)

        Y = self.simulate_F1F2()

        while True:
            maxVal, activeNeuronIdx = maxIgnoreNan(Y)
            if activeNeuronIdx == -1:
                raise Exception("No capacity left!")

            if self.check_cluster(activeNeuronIdx):
                return activeNeuronIdx
            else:
                Y[activeNeuronIdx] = nan
    
    def process(self, S, learn_mode):
        activeNeuronIdx = self.find_class(S)

        if (activeNeuronIdx == self.nbOfClasses):
            activeNeuronIdx = self.nbOfClasses
            self.nbOfClasses += 1             
        if learn_mode:
            self.learning(S, activeNeuronIdx)
        return activeNeuronIdx
    
    def get_cluster_exemplar(self, j):
        return self.f1f2[:, j]

    def log_description(self):
        print("Network description:")
        print(f'Properties:\n a = {self.a}, b = {self.b}, c = {self.c}, d = {self.d}, e = {self.e}')
        print(f'learningRate = {self.learningRate}')
        print(f'learningLength = {self.learningLength}')
        print(f'vigilance (rho) = {self.vigilance}')
        print(f'theta = {self.theta}')