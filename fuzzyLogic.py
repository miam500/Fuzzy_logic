import matplotlib.pyplot as plt
import numpy as np
import pygame
import skfuzzy as fuzz
from fcmeans import FCM
from skfuzzy import control as ctrl

from noise import *


def createFuzzyController():
    # TODO: add in
    mic0 = ctrl.Antecedent(np.linspace(0, 10), 'mic0')
    mic1 = ctrl.Antecedent(np.linspace(0, 10), 'mic1')
    mic2 = ctrl.Antecedent(np.linspace(0, 10), 'mic2')
    # TODO: add out
    x = ctrl.Consequent(np.linspace(-1, 1), 'x')
    y = ctrl.Consequent(np.linspace(-1, 1), 'y')
    # TODO: membership function
    names = ['VF', 'F', 'M', 'C', 'VC']
    mic0.automf(names=names)
    mic1.automf(names=names)
    mic2.automf(names=names)
    
    names = ['L', 'M', 'R']
    x.automf(names=names)
    names = ['U', 'M', 'D']
    y.automf(names=names)
    
    # TODO: define rules
    rules = []
    ##############################################
    #                   mic0                     #
    ##############################################
    rules.append(ctrl.Rule(antecedent=(mic0['VF'] & mic1[''] & mic2[''])))
    ##############################################
    #                   mic1                     #
    ##############################################
    ##############################################
    #                   mic2                     #
    ##############################################

    for rule in rules:
        rule.and_func = np.fmin
        rule.or_func = np.fmax

    system = ctrl.ControlSystem(rules=rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim


class Controller():
    def __init__(self,grid_cells,mics,sampling_grid = 2,membership_fcns = 5):
        self.n_membership = membership_fcns
        self.grid = grid_cells
        self.mics = mics
        self.size = 5
        self.color = pygame.Color('red')
        self.sampling_grid = sampling_grid

        sampling_data = []

        from map import TILE
        from player import Player


        dummy = Player((0,0))
        dummy.add_noise(Constant_Noise(40))

        grid = self.sampling_grid

        for col in grid_cells:
            col_data = []
            for cell in col:
                for ix in range(grid):
                    for iy in range(grid):
                        x = cell.x * TILE + (ix+1)*TILE/(grid+1)
                        y = cell.y * TILE + (iy+1)*TILE/(grid+1)
                        dummy.x, dummy.y = x,y
                        dummy.propagate_sound(grid_cells,mics,0)
                        col_data += [self.get_input()]
            sampling_data += col_data

        sampling_data = np.array(sampling_data)
        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data=sampling_data.T,c=membership_fcns,m=2,error=0.005, maxiter=1000, init=None)
        #self.fcm = FCM(n_clusters=membership_fcns)
        #self.fcm.fit(sampling_data)
        self.centers_i = np.array([cntr[:,0]])
        self.centers_j = np.array([cntr[:,1]])
        self.centers_k = np.array([cntr[:,2]])
        
        return           
    
    def get_pos_prediction(self):
        x,y = 0,0
        return x,y

    def get_input(self):
        input = []
        for mic in self.mics:
            input += [mic.data[-1]]
        return input

    def draw(self,sc):
        x,y = self.get_pos_prediction()
        print(x,y)
        pygame.draw.circle(sc,self.color,center=(x,y),radius=self.size)

class MSE_linear(Controller):
    def __init__(self, grid_cells, mics):
        super().__init__(grid_cells, mics)

        from map import TILE
        from player import Player


        self.phi = []
        self.Yx = []
        self.Yy = []
        dummy = Player((0,0))
        dummy.add_noise(Constant_Noise(40))
        grid = self.sampling_grid
        for col in grid_cells:
            col_phi = []
            for cell in col:
                for ix in range(grid):
                    for iy in range(grid):
                        x = cell.x * TILE + ix*TILE/(grid+1)
                        y = cell.y * TILE + iy*TILE/(grid+1)
                        dummy.x, dummy.y = x,y
                        dummy.propagate_sound(grid_cells,mics,0)
                        col_phi += [self.get_input()]
                        self.Yx += [x]
                        self.Yy += [y]

            if len(self.phi):
                self.phi = np.vstack((self.phi,col_phi))
            else:
                self.phi = col_phi

        inv = np.linalg.inv(self.phi.T@self.phi)@self.phi.T
        self.theta_x = inv@self.Yx
        self.theta_y = inv@self.Yy
        
        for mic in mics:
            mic.data = []
            mic.t = []

    
    def get_pos_prediction(self):
        x = self.get_prediction(self.theta_x)
        y = self.get_prediction(self.theta_y)
        return x,y

    def get_prediction(self,theta):
        input = np.array(self.get_input())
        out = np.matmul(theta.T,input.T)
        return out


class MSE_Fuzzy(Controller):
    def __init__(self, grid_cells, mics):
        super().__init__(grid_cells, mics)

        from map import TILE
        from player import Player

        
        self.phi = []
        self.Yx = []
        self.Yy = []
        dummy = Player((0,0))
        dummy.add_noise(Constant_Noise(40))
        grid = self.sampling_grid
        for col in grid_cells:
            col_phi = []
            for cell in col:
                for ix in range(grid):
                    for iy in range(grid):
                        x = cell.x * TILE + ix*TILE/(grid+1)
                        y = cell.y * TILE + iy*TILE/(grid+1)
                        dummy.x, dummy.y = x,y
                        dummy.propagate_sound(grid_cells,mics,0)
                        col_phi += [self.get_eps()]
                        self.Yx += [x]
                        self.Yy += [y]

            if len(self.phi):
                self.phi = np.vstack((self.phi,col_phi))
            else:
                self.phi = col_phi

        inv = np.linalg.inv(self.phi.T@self.phi)@self.phi.T
        self.theta_x = inv@self.Yx
        self.theta_y = inv@self.Yy

        clip = False
        if clip:
            self.theta_x = np.clip(self.theta_x,0,900)
            self.theta_y = np.clip(self.theta_y,0,900)
        
        for mic in mics:
            mic.data = []
            mic.t = []
    
    def get_eps(self):
        eps = np.zeros(self.n_membership**3)
        mic_i,mic_j,mic_k = self.get_input()
        mic_i = np.array([[mic_i]])
        mic_j = np.array([[mic_j]])
        mic_k = np.array([[mic_k]])
        #out = self.fcm.predict(in_data)
        #i,j,k = out
        #eps[i*self.n_membership**2+j*self.n_membership+k] = 1
        total_mem = 0
        mem_i, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(test_data=mic_i,cntr_trained = self.centers_i.T,m=2, error=0.005, maxiter=1000,init = None,seed=None)
        mem_j, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(test_data=mic_j,cntr_trained = self.centers_j.T,m=2, error=0.005, maxiter=1000,init = None,seed=None)
        mem_k, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(test_data=mic_k,cntr_trained = self.centers_k.T,m=2, error=0.005, maxiter=1000,init = None,seed=None)
                        
        for i in range(self.n_membership):
            for j in range(self.n_membership):
                for k in range(self.n_membership):
                    
                    mem = mem_i[i]*mem_j[j]*mem_k[k]
                    eps[i*self.n_membership**2+j*self.n_membership+k] = mem
                    total_mem += mem


        
        if total_mem != 0:
            eps = eps/total_mem
        else:
            raise Exception("")
        return eps

    def get_pos_prediction(self):
        x = self.get_prediction(self.theta_x)
        y = self.get_prediction(self.theta_y)
        return x,y

    def get_prediction(self,theta):
        eps = self.get_eps()
        out = np.matmul(theta.T,eps.T)
        return out

class test(Controller):
    def __init__(self, grid_cells, mics, sampling_grid=10, membership_fcns=5):
        super().__init__(grid_cells, mics, sampling_grid, membership_fcns)
        

        colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

        # Define three cluster centers
        centers = [[4, 2],
                [1, 7],
                [5, 6]]

        # Define three cluster sigmas in x and y, respectively
        sigmas = [[0.8, 0.3],
                [0.3, 0.5],
                [1.1, 0.7]]

        # Generate test data
        np.random.seed(42)  # Set seed for reproducibility
        xpts = np.zeros(1)
        ypts = np.zeros(1)
        labels = np.zeros(1)
        for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
            xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
            ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
            labels = np.hstack((labels, np.ones(200) * i))

        # Visualize the test data
        fig0, ax0 = plt.subplots()
        for label in range(3):
            ax0.plot(xpts[labels == label], ypts[labels == label], '.',
                    color=colors[label])
        ax0.set_title('Test data: 200 points x3 clusters.')
        # Set up the loop and plot
        fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
        alldata = np.vstack((xpts, ypts))
        fpcs = []

        for ncenters, ax in enumerate(axes1.reshape(-1), 2):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

            # Store fpc values for later
            fpcs.append(fpc)

            # Plot assigned clusters, for each data point in training set
            cluster_membership = np.argmax(u, axis=0)
            for j in range(ncenters):
                ax.plot(xpts[cluster_membership == j],
                        ypts[cluster_membership == j], '.', color=colors[j])

            # Mark the center of each fuzzy cluster
            for pt in cntr:
                ax.plot(pt[0], pt[1], 'rs')

            ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
            ax.axis('off')

        fig1.tight_layout()

        # Regenerate fuzzy model with 3 cluster centers - note that center ordering
        # is random in this clustering algorithm, so the centers may change places
        cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
            alldata, 3, 2, error=0.005, maxiter=1000)

        # Show 3-cluster model
        fig2, ax2 = plt.subplots()
        ax2.set_title('Trained model')
        for j in range(3):
            ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
                    alldata[1, u_orig.argmax(axis=0) == j], 'o',
                    label='series ' + str(j))
        ax2.legend()

        # Generate uniformly sampled data spread across the range [0, 10] in x and y
        newdata = np.random.uniform(0, 1, (1100, 2)) * 10

        # Predict new cluster membership with `cmeans_predict` as well as
        # `cntr` from the 3-cluster model
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            newdata.T, cntr, 2, error=0.005, maxiter=1000)

        # Plot the classified uniform data. Note for visualization the maximum
        # membership value has been taken at each point (i.e. these are hardened,
        # not fuzzy results visualized) but the full fuzzy result is the output
        # from cmeans_predict.
        cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization

        fig3, ax3 = plt.subplots()
        ax3.set_title('Random points classifed according to known centers')
        for j in range(3):
            ax3.plot(newdata[cluster_membership == j, 0],
                    newdata[cluster_membership == j, 1], 'o',
                    label='series ' + str(j))
        ax3.legend()

        plt.show()