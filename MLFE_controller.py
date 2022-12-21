import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import scipy.io


class Controller_MLFE:
    def __init__(self, grid_cells, mics, data=None, theta=None):
        from fuzzyLogic_MLFE import DATAFILE, THETAFILE
        self.grid = grid_cells
        self.mics = mics
        if data is None:
            self.generate_data()
            data = scipy.io.loadmat(DATAFILE)
        if theta is None:
            self.obtain_parameters(data)
            theta = scipy.io.loadmat(THETAFILE)
        self.fc = self.fuzzy_ctrl(theta)

    # Obsolete, does not generate good data
    def generate_data(self):
        from map import TILE
        from player import Player
        from noise import Constant_Noise
        in_data = []
        outx = []
        outy = []
        dummy = Player((0, 0))
        dummy.add_noise(Constant_Noise(40))
        grid = 1
        for col in self.grid:
            col_data = []
            for cell in col:
                for ix in range(grid):
                    for iy in range(grid):
                        x = cell.x * TILE + (ix + 1) * TILE / (grid + 1)
                        y = cell.y * TILE + (iy + 1) * TILE / (grid + 1)
                        dummy.x, dummy.y = x, y
                        dummy.propagate_sound(self.grid, self.mics, 0)
                        outx += [x]
                        outy += [y]
                        col_data += [self.get_input()]
            if len(in_data):
                in_data = np.vstack((in_data, col_data))
            else:
                in_data = col_data
        data = {
            "in1": in_data[:, 0],
            "in2": in_data[:, 1],
            "in3": in_data[:, 2],
            "outx": outx,
            "outy": outy
        }
        scipy.io.savemat('iodata2.mat', data)

    def get_input(self):
        input = []
        for mic in self.mics:
            input += [mic.data[-1]]
        return input

    def obtain_parameters(self, data):
        # j: itérateur pour nombre d'entrées (0, 1, ..., n)
        # i: itérateur pour nombre de règles (0, 1, ..., R)
        # n: nombre d'entrées (3)
        # R: nombre de règles (R)
        # Load input and output data
        outx = data["outx"].T
        outy = data["outy"].T
        in1 = data["in1"].T
        in2 = data["in2"].T
        in3 = data["in3"].T

        # Initialize parameters
        R = 1  # Initial number of rules
        W = 1  # Weighting factor for membership functions overlap
        sigma0 = 0.5  # Initial spread of the input membership function
        epsilon = 200  # Tolerated error
        theta = {  # Initial fuzzy system parameters
            "bx": outx[0],
            "by": outy[0],
            "c1": in1[0],
            "c2": in2[0],
            "c3": in3[0],
            "sigma1": np.array([sigma0]),
            "sigma2": np.array([sigma0]),
            "sigma3": np.array([sigma0])
        }
        # Loop for all data "pairs"
        total_error = 0
        for k in range(1, len(in1) - 1):
            # Initialize numerators and denominators
            numx = 0
            denomx = 0
            numy = 0
            denomy = 0
            # Loop for all rules
            for i in range(0, R):
                # Calculate numerators and denominators
                numx += theta["bx"][i] * \
                        np.exp((-1 / 2) * ((in1[k, 0] - theta["c1"][i]) / theta["sigma1"][i]) ** 2) * \
                        np.exp((-1 / 2) * ((in2[k, 0] - theta["c2"][i]) / theta["sigma2"][i]) ** 2) * \
                        np.exp((-1 / 2) * ((in3[k, 0] - theta["c3"][i]) / theta["sigma3"][i]) ** 2)
                denomx += np.exp((-1 / 2) * ((in1[k, 0] - theta["c1"][i]) / theta["sigma1"][i]) ** 2) * \
                          np.exp((-1 / 2) * ((in2[k, 0] - theta["c2"][i]) / theta["sigma2"][i]) ** 2) * \
                          np.exp((-1 / 2) * ((in3[k, 0] - theta["c3"][i]) / theta["sigma3"][i]) ** 2)
                numy += theta["by"][i] * \
                        np.exp((-1 / 2) * ((in1[k, 0] - theta["c1"][i]) / theta["sigma1"][i]) ** 2) * \
                        np.exp((-1 / 2) * ((in2[k, 0] - theta["c2"][i]) / theta["sigma2"][i]) ** 2) * \
                        np.exp((-1 / 2) * ((in3[k, 0] - theta["c3"][i]) / theta["sigma3"][i]) ** 2)
                denomy += np.exp((-1 / 2) * ((in1[k, 0] - theta["c1"][i]) / theta["sigma1"][i]) ** 2) * \
                          np.exp((-1 / 2) * ((in2[k, 0] - theta["c2"][i]) / theta["sigma2"][i]) ** 2) * \
                          np.exp((-1 / 2) * ((in3[k, 0] - theta["c3"][i]) / theta["sigma3"][i]) ** 2)
            # Calculate approximation for x and y
            fx = numx / denomx
            fy = numy / denomy
            if np.isnan(fx):
                fx = 0.0
            if np.isnan(fy):
                fy = 0.0
            # If system represents data well enough, go to next data "pair"
            error = np.abs((fx - outx[k])) + np.abs((fy - outy[k]))
            total_error += error
            if error < epsilon:
                continue
            else:
                # Create new rule and modify membership functions
                R += 1
                theta["bx"] = np.append(theta["bx"], outx[k])
                theta["by"] = np.append(theta["by"], outy[k])
                theta["c1"] = np.append(theta["c1"], in1[k, 0])
                theta["c2"] = np.append(theta["c2"], in2[k, 0])
                theta["c3"] = np.append(theta["c3"], in3[k, 0])
                abs1 = np.abs(theta["c1"][0:len(theta["c1"])] - in1[k, 0])
                abs2 = np.abs(theta["c2"][0:len(theta["c2"])] - in2[k, 0])
                abs3 = np.abs(theta["c3"][0:len(theta["c3"])] - in3[k, 0])
                if abs1[abs1 != 0].size == 0 or (abs1-0.001)[(abs1-0.001) < 0.0].size > 0:
                    theta["sigma1"] = np.append(theta["sigma1"], sigma0)
                else:
                    argmin1 = np.argmin(abs1[abs1 != 0])
                    theta["sigma1"] = np.append(theta["sigma1"], (1 / W) * np.abs(in1[k, 0] - theta["c1"][argmin1]))
                if abs2[abs2 != 0].size == 0 or (abs2-0.001)[(abs2-0.001) < 0.0].size > 0:
                    theta["sigma2"] = np.append(theta["sigma2"], sigma0)
                else:
                    argmin2 = np.argmin(abs2[abs2 != 0])
                    theta["sigma2"] = np.append(theta["sigma2"], (1 / W) * np.abs(in2[k, 0] - theta["c2"][argmin2]))
                if abs3[abs3 != 0].size == 0 or (abs3-0.001)[(abs3-0.001) < 0.0].size > 0:
                    theta["sigma3"] = np.append(theta["sigma3"], sigma0)
                else:
                    argmin3 = np.argmin(abs3[abs3 != 0])
                    theta["sigma3"] = np.append(theta["sigma3"], (1 / W) * np.abs(in3[k, 0] - theta["c3"][argmin3]))

        scipy.io.savemat('fcparamscustom2.mat', theta)

    def fuzzy_ctrl(self, theta):
        # Retrieve parameters
        R = np.size(theta["sigma1"])
        sigma0 = theta["sigma1"][0, 0]

        # Create universes for inputs and outputs
        mic1 = ctrl.Antecedent(np.linspace(0, 60, 1001), 'mic1')
        mic2 = ctrl.Antecedent(np.linspace(0, 60, 1001), 'mic2')
        mic3 = ctrl.Antecedent(np.linspace(0, 60, 1001), 'mic3')
        posx = ctrl.Consequent(np.linspace(0, 1000, 1001), 'posx')
        posy = ctrl.Consequent(np.linspace(0, 1000, 1001), 'posy')

        sigout = 25  # Width of output membership functions (custom parameter)
        for i in range(R):
            mic1[str(i)] = fuzz.gaussmf(mic1.universe, theta["c1"][0, i], theta["sigma1"][0, i])
            mic2[str(i)] = fuzz.gaussmf(mic2.universe, theta["c2"][0, i], theta["sigma2"][0, i])
            mic3[str(i)] = fuzz.gaussmf(mic3.universe, theta["c3"][0, i], theta["sigma3"][0, i])
            posx[str(i)] = fuzz.gaussmf(posx.universe, theta["bx"][0, i], sigout)
            posy[str(i)] = fuzz.gaussmf(posy.universe, theta["by"][0, i], sigout)

        # mic1.view()
        # mic2.view()
        # mic3.view()
        # posx.view()
        # posy.view()
        # plt.show()

        # Rules
        rules = []
        for i in range(R):
            rules = np.append(rules, ctrl.Rule((mic1[str(i)] & mic2[str(i)] & mic3[str(i)]), (posx[str(i)], posy[str(i)])))

        pos_ctrl = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(pos_ctrl)
        return sim
