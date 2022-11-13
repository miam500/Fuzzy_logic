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
    def __init__(self,grid_cells,mics):
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
        pygame.draw.circle(sc,self.color,center=(x,y),radius=self.size)

# Basé sur le ppt chapitre #5, pt mal implementé
class MSE(Controller):
    def __init__(self, grid_cells, mics):
        super().__init__(grid_cells, mics)

        from map import TILE
        from player import Player

        
        self.phi = []
        self.Yx = []
        self.Yy = []
        dummy = Player((0,0))
        dummy.add_noise(Constant_Noise(40))
        for row in grid_cells:
            row_phi = []
            for cell in row:
                x = cell.x * TILE + TILE/2
                y = cell.y * TILE + TILE/2
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
        

    
    def get_pos_prediction(self):
        x = self.get_prediction(self.theta_x)
        y = self.get_prediction(self.theta_y)
        return x,y

    def get_prediction(self,theta):
        input = np.array(self.get_input())
        out = np.matmul(theta.T,input.T)
        return out

        