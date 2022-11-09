import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

def createFuzzyController():
    # TODO: add in
    mic_0 = ctrl.Antecedent(np.linspace(0, 10), 'mic_0')
    mic_1 = ctrl.Antecedent(np.linspace(0, 10), 'mic_1')
    mic_2 = ctrl.Antecedent(np.linspace(0, 10), 'mic_2')
    # TODO: add out
    x = ctrl.Consequent(np.linspace(0, 1), 'x')
    y = ctrl.Consequent(np.linspace(0, 1), 'y')
    
    # TODO: membership function
    
    # TODO: define rules
    rules = []
    rules.append(ctrl.Rule()) # ex
    
    for rule in rules:
        rule.and_func = np.fmin
        rule.or_func = np.fmax
    
    system = ctrl.ControlSystem(rules=rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim
