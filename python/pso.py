
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high):
       
        self.dimension = dimension 
        self.time = time 
        self.size = size  
        self.bound = []  
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension)) 
        self.v = np.zeros((self.size, self.dimension))  
        self.p_best = np.zeros((self.size, self.dimension))  
        self.g_best = np.zeros((1, self.dimension))[0]  
       
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]  
            fit = self.fitness(self.p_best[i])
            
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, x):
        
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        y = math.floor((x2 * np.exp(x1) + x3 * np.sin(x2) + x4 + x5) * 100) / 100
        # print(y)
        return y

    def update(self, size):
        c1 = 2.0  
        c2 = 2.0
        w = 0.8  
        for i in range(size):
           
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

           
            self.x[i] = self.x[i] + self.v[i]
            
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            
            if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) > self.fitness(self.g_best):
                self.g_best = self.x[i]

    def pso(self):
        best = []
        self.final_best = np.array([1, 2, 3, 4, 5])
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            print('Current best position：{}'.format(self.final_best))
            temp = self.fitness(self.final_best)
            print('Current best fit：{}'.format(temp))
            best.append(temp)
        t = [i for i in range(self.time)]
        plt.figure()
        plt.plot(t, best, color='red', marker='.', ms=15)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"Number of iterations") 
        plt.ylabel(u"degree of adaptation")  
        plt.title(u"iterative process")  
        plt.show()


if __name__ == '__main__':
    time = 50
    size = 100
    dimension = 5
    v_low = -1
    v_high = 1
    low = [1, 1, 1, 1, 1]
    up = [25, 25, 25, 25, 25]
    pso = PSO(dimension, time, size, low, up, v_low, v_high)
    pso.pso()
