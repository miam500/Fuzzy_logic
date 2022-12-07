import math
import numpy as np
from copy import deepcopy

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

    def propagate_sound(self,pos,grid_cells,mics,dist = 0,come_from =  None, previous_cells = []):
        
        from map import TILE
        x,y = pos
        ix = int((x) // TILE)
        iy = int((y) // TILE)


        if [ix,iy] in previous_cells:
            return
        else:
            previous_cells += [[ix,iy]]

        cell = grid_cells[ix][iy]

        if dist < cell.dist +90:

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
                    self.propagate_sound(pos,grid_cells,mics,dist+dist_to_left,'left',deepcopy(previous_cells))
                if not cell.walls['right'] and i == 1 and not come_from == 'left':
                    pos = (x+dist_to_right+1, y)
                    self.propagate_sound(pos,grid_cells,mics,dist+dist_to_right,'right',deepcopy(previous_cells))
                if not cell.walls['top']  and i == 2 and not come_from == 'bot':
                    pos = (x, y - dist_to_top - 1)
                    self.propagate_sound(pos,grid_cells,mics,dist+dist_to_top,'top',deepcopy(previous_cells))
                if not cell.walls['bot']  and i == 3 and not come_from == 'top':
                    pos = (x, y + dist_to_bot + 1) 
                    self.propagate_sound(pos,grid_cells,mics,dist+dist_to_bot,'bot',deepcopy(previous_cells))

    def propagate_sound_dijkstra(self,pos,grid_cells,mics,dist = 99999):
        
        from map import TILE
        pixels = np.ones(shape=(TILE*len(grid_cells),TILE*len(grid_cells[0]))) * dist
        dist = 0
        x,y = int(pos[0]),int(pos[1])
        pixels[x][y] = dist

        while True:
            pixels_to_propagate_x,pixels_to_propagate_y = np.where(pixels == dist)
            dist += 1
            if len(pixels_to_propagate_x) > 0:
                for i in range(len(pixels_to_propagate_x)):
                    x,y = pixels_to_propagate_x[i],pixels_to_propagate_y[i]
                    self.check_pixel_dijkstra(x,y,grid_cells,pixels,dist)
            else:
                return
        


    def check_pixel_dijkstra(self,x,y,grid_cells,pixels,dist):
        pixels_to_check = [[x+1,y],[x-1,y],[x,y+1],[x,y-1]]
        for x,y in pixels_to_check:
            if  0 <= x < 900 and 0 <= y < 900 and pixels[x,y] > dist:
                pixels[x,y] = dist

    #https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
    def propagate_sound_astar(self,pos,grid_cells,t,mics,factor=20):
        from map import TILE

        for mic in mics:
            x,y = int(pos[0]//factor),int(pos[1]//factor+1)
            goal = (mic.x//factor,mic.y//factor)
            closed_list = []
            if x == 0:
                x+=1
            if y == 0:
                y+=1
            open_list = [self.Node(x,y,1,goal)]

            while open_list:
                open_list.sort()
                current_pixel = open_list[0]
                x,y,g = current_pixel.x ,current_pixel.y,current_pixel.g
                open_list.remove(current_pixel)
                closed_list.append(current_pixel)

                if (x,y) == goal:
                    dist = closed_list[-1].g*factor
                    mic.record(self.get_apparent_sound(t,dist),t)
                    break
                
                if x % (TILE/factor) == 0:
                    ix = int((x*factor) // TILE)
                    iy = int((y*factor) // TILE)
                    try:
                        cell = grid_cells[ix][iy]
                        if cell.walls['left']:
                            continue
                    except:
                        continue

                if y % (TILE/factor) == 0:
                    ix = int((x*factor) // TILE)
                    iy = int((y*factor) // TILE)
                    try:
                        cell = grid_cells[ix][iy]
                        if cell.walls['top']:
                            continue
                    except:
                        continue
                
                
                current_pixel.children = [self.Node(x+1,y,g+1,goal),self.Node(x-1,y,g+1,goal),self.Node(x,y+1,g+1,goal),self.Node(x,y-1,g+1,goal)]

                for child in current_pixel.children:
                    if child in closed_list:
                        continue

                    if child in open_list:
                        previous_child = open_list[open_list.index(child)]
                        if previous_child.g  <= child.g:
                            continue
                        else:
                            pass
                    
                    open_list.append(child)

    
    class Node():
        def __init__(self,x,y,g,goal) -> None:
            self.x = x
            self.y = y
            self.g = g
            self.h = (goal[0]-x)**2 + (goal[1]-y)**2
            self.f = self.g + self.h

            self.children = None
            return
        
        def __eq__(self,other):
            return self.x == other.x and self.y == other.y

        def __lt__(self,other):
            return self.f < other.f
        
        def __repr__(self):
            return "x="+str(self.x) +", y="+ str(self.y) +", f="+ str(self.f)
            
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

