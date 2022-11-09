import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

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
