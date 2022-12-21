import pygame
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from map import *

class createFuzzyController():
    def __init__(self, grid_cells, mic, sampling_grid=2, membership_fcns=5):
        self.n_membership = membership_fcns
        self.grid = grid_cells
        self.mic = mic
        self.size = 30
        self.color = pygame.Color('red')
        self.sampling_grid = sampling_grid

        # TODO: add in
        mic = ctrl.Antecedent(np.linspace(0, 60), 'mic')
        x = ctrl.Antecedent(np.linspace(0, WIDTH), 'x')
        y = ctrl.Antecedent(np.linspace(0, HEIGHT), 'y')
        # TODO: add out
        x_out = ctrl.Consequent(np.linspace(0, WIDTH), 'x_out')
        y_out = ctrl.Consequent(np.linspace(0, HEIGHT), 'y_out')
        # TODO: membership function
        names = ['F', 'M', 'C']
        mic.automf(names=names)
        names = ['L', 'R']
        x.automf(names=names)
        names = ['U', 'D']
        y.automf(names=names)

        #names = ['LL', 'L', 'R', 'RR']
        #x_out.automf(names=names)
        x_out['L'] = fuzz.trimf(x_out.universe, [0, 0, self.mic.x])
        x_out['M'] = fuzz.trimf(x_out.universe, [self.mic.x/2, self.mic.x, self.mic.x+(WIDTH - self.mic.x)/2])
        x_out['R'] = fuzz.trimf(x_out.universe, [self.mic.x, WIDTH, WIDTH])
        #names = ['UU', 'U', 'D', 'DD']
        #y_out.automf(names=names)
        y_out['U'] = fuzz.trimf(y_out.universe, [0, 0, self.mic.y])
        y_out['M'] = fuzz.trimf(y_out.universe, [self.mic.y/2, self.mic.y, self.mic.y + (HEIGHT - self.mic.y)/2])
        y_out['D'] = fuzz.trimf(y_out.universe, [self.mic.y, HEIGHT, HEIGHT])

        # TODO: define rules
        rules = []
        ##############################################
        #                   mic F                    #
        ##############################################
        rules.append(ctrl.Rule(antecedent=(mic['F'] & x['L']), consequent=(x_out['R'])))
        rules.append(ctrl.Rule(antecedent=(mic['F'] & y['U']), consequent=(y_out['D'])))

        rules.append(ctrl.Rule(antecedent=(mic['F'] & x['L']), consequent=(x_out['R'])))
        rules.append(ctrl.Rule(antecedent=(mic['F'] & y['D']), consequent=(y_out['U'])))

        rules.append(ctrl.Rule(antecedent=(mic['F'] & x['R']), consequent=(x_out['L'])))
        rules.append(ctrl.Rule(antecedent=(mic['F'] & y['U']), consequent=(y_out['D'])))

        rules.append(ctrl.Rule(antecedent=(mic['F'] & x['R']), consequent=(x_out['L'])))
        rules.append(ctrl.Rule(antecedent=(mic['F'] & y['D']), consequent=(y_out['U'])))

        ##############################################
        #                   mic M                    #
        ##############################################
        rules.append(ctrl.Rule(antecedent=(mic['M'] & x['L']), consequent=(x_out['R'])))
        rules.append(ctrl.Rule(antecedent=(mic['M'] & y['U']), consequent=(y_out['D'])))

        rules.append(ctrl.Rule(antecedent=(mic['M'] & x['L']), consequent=(x_out['R'])))
        rules.append(ctrl.Rule(antecedent=(mic['M'] & y['D']), consequent=(y_out['U'])))

        rules.append(ctrl.Rule(antecedent=(mic['M'] & x['R']), consequent=(x_out['L'])))
        rules.append(ctrl.Rule(antecedent=(mic['M'] & y['U']), consequent=(y_out['D'])))

        rules.append(ctrl.Rule(antecedent=(mic['M'] & x['R']), consequent=(x_out['L'])))
        rules.append(ctrl.Rule(antecedent=(mic['M'] & y['D']), consequent=(y_out['U'])))
        ##############################################
        #                   mic C                    #
        ##############################################
        rules.append(ctrl.Rule(antecedent=(mic['C'] & x['L']), consequent=(x_out['M'])))
        rules.append(ctrl.Rule(antecedent=(mic['C'] & y['U']), consequent=(y_out['M'])))

        rules.append(ctrl.Rule(antecedent=(mic['C'] & x['L']), consequent=(x_out['M'])))
        rules.append(ctrl.Rule(antecedent=(mic['C'] & y['D']), consequent=(y_out['M'])))

        rules.append(ctrl.Rule(antecedent=(mic['C'] & x['R']), consequent=(x_out['M'])))
        rules.append(ctrl.Rule(antecedent=(mic['C'] & y['U']), consequent=(y_out['M'])))

        rules.append(ctrl.Rule(antecedent=(mic['C'] & x['R']), consequent=(x_out['M'])))
        rules.append(ctrl.Rule(antecedent=(mic['C'] & y['D']), consequent=(y_out['M'])))

        for rule in rules:
            rule.and_func = np.fmin
            rule.or_func = np.fmax

        system = ctrl.ControlSystem(rules=rules)
        sim = ctrl.ControlSystemSimulation(system)

        self.sim = sim

    def get_input(self):
        return self.mic.data
