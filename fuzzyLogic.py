import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

def createFuzzyController():
    # TODO: add in

    # TODO: add out
    
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
