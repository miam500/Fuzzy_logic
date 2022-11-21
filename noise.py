import math
import numpy as np

class Noise:
    def __init__(self,dB=0):
        self.dB = dB
    
    def get_amplitude(self,t):
        return 0
    
    def get_apparent_sound(self,t,pixels):
        from map import TILE
        distance = pixels / TILE
        amp = self.get_amplitude(t)
        dB = amp - 20*math.log10(distance/1)
        return  dB

    def propagate_sound(self,pos,grid_cells,mics,t,dist = 0,come_from =  None):
        
        from map import TILE
        x,y = pos
        ix = int((x) // TILE)
        iy = int((y) // TILE)

        cell = grid_cells[ix][iy]
        if dist < cell.dist +50:

            if dist < cell.dist:
                cell.dist = dist
                cell.pos = pos

            dist_to_left =  x - (ix * TILE)
            dist_to_right =  ((ix+1) * TILE) - x
            dist_to_top =  y - (iy * TILE)
            dist_to_bot =  ((iy+1) * TILE) - y

            order = np.argsort([dist_to_left,dist_to_right,dist_to_top,dist_to_bot])
            for i in order:
                if not cell.walls['left'] and i == 0 and not come_from == 'right':
                    pos = (x-dist_to_left-1, y)
                    self.propagate_sound(pos,grid_cells,mics,t,dist+dist_to_left,'left')
                if not cell.walls['right'] and i == 1 and not come_from == 'left':
                    pos = (x+dist_to_right+1, y)
                    self.propagate_sound(pos,grid_cells,mics,t,dist+dist_to_right,'right')
                if not cell.walls['top']  and i == 2 and not come_from == 'bot':
                    pos = (x, y - dist_to_top - 1)
                    self.propagate_sound(pos,grid_cells,mics,t,dist+dist_to_top,'top')
                if not cell.walls['bot']  and i == 3 and not come_from == 'top':
                    pos = (x, y + dist_to_bot + 1) 
                    self.propagate_sound(pos,grid_cells,mics,t,dist+dist_to_bot,'bot')


class Constant_Noise(Noise):
    
    def __init__(self,dB):
        self.dB = dB

    def get_amplitude(self,t):
        return self.dB

class Sine_Noise(Noise):
    
    def __init__(self,dB,w=math.pi):
        self.dB = dB
        self.w = w

    def get_amplitude(self,t):
        return 10 * math.sin(self.w*t) + self.dB

