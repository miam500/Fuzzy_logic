import itertools
import matplotlib.pyplot as plt
import numpy as np
import pygame
import skfuzzy as fuzz
from fcmeans import FCM
from skfuzzy import control as ctrl

from noise import *



class Controller():
    def __init__(self,grid_cells,mics,sampling_grid =5,membership_fcns = 5):
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

        print("Generating Fuzzy clustering")
        print('')
        total = len(grid_cells)*len(grid_cells[0])*grid*grid
        for i, col in enumerate(grid_cells):
            col_data = []
            for j, cell in enumerate(col):
                for ix in range(grid):
                    for iy in range(grid):
                        progress = 100*(i*len(grid_cells[0])*grid**2+j*grid**2+ix*grid+iy)/total
                        print(f"\rProgress:{progress} %")
                        x = cell.x * TILE + (ix+1)*TILE/(grid+1)
                        y = cell.y * TILE + (iy+1)*TILE/(grid+1)
                        dummy.x, dummy.y = x,y
                        dummy.propagate_sound(grid_cells,mics,0)
                        col_data += [self.get_input()]
            sampling_data += col_data

        sampling_data = np.array(sampling_data)
        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data=sampling_data.T,c=membership_fcns,m=2,error=0.005, maxiter=1000, init=None)
        
        self.centers = {}
        for i in range(len(self.get_input())):
            key = f"center_{i}"
            self.centers[key] = np.array([cntr[:, i]])
        
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
        #print(x,y)
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
        mics = {}
        for i, val in enumerate(self.get_input()):
            key = f"mic_{i}"
            mics[key] = np.array([[val]])
        eps = np.zeros(self.n_membership ** len(mics))

        mem = []
        u = []
        d =[]
        jm = []
        p =[]
        fpc =[]
        for i, mic in enumerate(mics.values()):
            mem_i, u_i, d_i, jm_i, p_i, fpc_i = fuzz.cluster.cmeans_predict(test_data=mic,
                                                                   cntr_trained=self.centers[f"center_{i}"].T, m=2,
                                                                   error=0.005, maxiter=1000, init=None, seed=None)
            mem.append(mem_i)
            u.append(u_i)
            d.append(d_i)
            jm.append(jm_i)
            p.append(p_i)
            fpc.append(fpc_i)

        total_mem = 0
        for_loop = []
        for i in range(len(mics)):
            for_loop.append(self.step_array(self.n_membership))
        for i, comb in enumerate(itertools.product(*for_loop)):
            mem_temp = 1
            for j in range(len(comb)):
                mem_temp = mem_temp * mem[j][comb[j]]
            eps[i] = mem_temp
            total_mem += mem_temp

        if total_mem != 0:
            eps = eps / total_mem
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
    
    def step_array(self, len_of):
        arr = []
        for i in range(len_of):
            arr.append(i)
        return arr

