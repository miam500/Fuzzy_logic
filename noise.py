import math

class Noise:
    def __init__(self,dB=0):
        self.dB = dB
    
    def get_amplitude(self,t):
        return 0
    
    def get_apparent_sound(self,t,pixels):
        from map import TILE
        distance = pixels / TILE
        dB = self.get_amplitude(t)
        return  dB / (distance**2)

    def propagate_sound(self,pos,grid_cells,mics,t,dist = 0):
        
        from map import TILE
        x,y = pos
        ix = int((x) // TILE)
        iy = int((y) // TILE)

        cell = grid_cells[ix][iy]
        if cell.dist > dist:

            cell.dist = dist
            cell.pos = pos

            dist_to_left =  x - (ix * TILE)
            dist_to_right =  ((ix+1) * TILE) - x
            dist_to_top =  y - (iy * TILE)
            dist_to_bot =  ((iy+1) * TILE) - y

            if not cell.walls['left']:
                pos = (x-dist_to_left-1, y)
                dist += dist_to_left
                self.propagate_sound(pos,grid_cells,mics,t,dist)
            if not cell.walls['right']:
                pos = (x+dist_to_right+1, y)
                dist += dist_to_right
                self.propagate_sound(pos,grid_cells,mics,t,dist)
            if not cell.walls['top']:
                pos = (x, y - dist_to_top - 1)
                dist += dist_to_top
                self.propagate_sound(pos,grid_cells,mics,t,dist)
            if not cell.walls['bot']:
                pos = (x, y + dist_to_bot + 1) 
                dist += dist_to_bot
                self.propagate_sound(pos,grid_cells,mics,t,dist)


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
        return self.dB * math.sin(self.w*t)

