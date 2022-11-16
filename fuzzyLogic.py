import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import pygame
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
    def __init__(self,grid_cells,mics,membership_fcns = 3):
        self.n_membership = membership_fcns
        self.grid = grid_cells
        self.mics = mics
        self.size = 5
        self.color = pygame.Color('red')
    
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
        grid = 10
        for row in grid_cells:
            row_phi = []
            for cell in row:
                for ix in range(grid):
                    for iy in range(grid):
                        x = cell.x * TILE + ix*TILE/(grid+1)
                        y = cell.y * TILE + iy*TILE/(grid+1)
                        dummy.x, dummy.y = x,y
                        dummy.propagate_sound(grid_cells,mics,0)
                        row_phi += [self.get_input()]
                        self.Yx += [x]
                        self.Yy += [y]

            if len(self.phi):
                self.phi = np.vstack((self.phi,row_phi))
            else:
                self.phi = row_phi

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


        min_dB = 10
        max_dB = 60
        interval = (max_dB - min_dB)/(self.n_membership-1)

        self.universe = np.linspace(min_dB, max_dB,100)

        mic0 = ctrl.Antecedent(self.universe, 'mic0')
        mic1 = ctrl.Antecedent(self.universe, 'mic1')
        mic2 = ctrl.Antecedent(self.universe, 'mic2')

        from map import RESOLUTION
        x = ctrl.Consequent(np.linspace(0, RESOLUTION[0]), 'x')
        y = ctrl.Consequent(np.linspace(0, RESOLUTION[1]), 'y')

        names = ['VF', 'F', 'M', 'C', 'VC']
        
        #self.mfs = [fuzz.trimf(self.universe,[min_dB - interval + i*interval,min_dB + i*interval,min_dB + interval + i*interval]) for i in range(self.n_membership)]
        self.mfs = [fuzz.trapmf(self.universe,[min_dB - interval + i*interval,min_dB-interval/2 + i*interval,min_dB+interval/2 + i*interval,min_dB + interval + i*interval]) for i in range(self.n_membership)]
        
        
        names = ['L', 'M', 'R']
        x.automf(names=names)
        names = ['U', 'M', 'D']
        y.automf(names=names)
        
        self.phi = []
        self.Yx = []
        self.Yy = []
        dummy = Player((0,0))
        dummy.add_noise(Constant_Noise(40))
        grid = 10
        for row in grid_cells:
            row_phi = []
            for cell in row:
                for ix in range(grid):
                    for iy in range(grid):
                        x = cell.x * TILE + ix*TILE/(grid+1)
                        y = cell.y * TILE + iy*TILE/(grid+1)
                        dummy.x, dummy.y = x,y
                        dummy.propagate_sound(grid_cells,mics,0)
                        row_phi += [self.get_eps()]
                        self.Yx += [x]
                        self.Yy += [y]

            if len(self.phi):
                self.phi = np.vstack((self.phi,row_phi))
            else:
                self.phi = row_phi

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
        total_mem = 0
        for i in range(self.n_membership):
            mem_i = fuzz._fuzzymath.interp_membership(self.universe,self.mfs[i],mic_i,False)
            for j in range(self.n_membership):
                mem_j = fuzz._fuzzymath.interp_membership(self.universe,self.mfs[j],mic_j,False)
                for k in range(self.n_membership):
                    mem_k = fuzz._fuzzymath.interp_membership(self.universe,self.mfs[k],mic_k,False)
                    mem = mem_i+mem_j+mem_k
                    eps[i*self.n_membership**2+j*self.n_membership+k] = mem
                    total_mem += mem
       
        if total_mem != 0:
            eps = eps/total_mem
        else:
            eps = np.zeros(self.n_membership**3)
        return eps

    def get_pos_prediction(self):
        x = self.get_prediction(self.theta_x)
        y = self.get_prediction(self.theta_y)
        return x,y

    def get_prediction(self,theta):
        eps = self.get_eps()
        out = np.matmul(theta.T,eps.T)
        return out
