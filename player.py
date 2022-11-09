import pygame
from noise import*


class Player:
    def __init__(self, pos, size = 16):
        x,y = pos
        self.rect = pygame.Rect(x-size/2,y-size/2,size,size)
        self.noise = Noise()
        self.x, self.y = x,y
        self.size = size
        self.color = (250,120,60)
        self.velx = 0 
        self.vely = 0
        self.acc = 4
        self.dec = 2
        self.max_speed = 20

        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False
    
    def add_noise(self,noise):
        self.noise = noise

    def draw(self,sc):
        self.rect = pygame.Rect(self.x - self.size/2,self.y - self.size/2,self.size,self.size)
        pygame.draw.rect(sc,self.color,self.rect)

    def move(self,key):
        if key == pygame.K_LEFT:
            self.move_left= True
        if key == pygame.K_RIGHT:
            self.move_right= True
        if key == pygame.K_UP:
            self.move_up= True
        if key == pygame.K_DOWN:
            self.move_down= True

    def stop_moving(self,key):
        if key == pygame.K_LEFT:
            self.move_left= False
        if key == pygame.K_RIGHT:
            self.move_right= False
        if key == pygame.K_UP:
            self.move_up= False
        if key == pygame.K_DOWN:
            self.move_down= False

    def check_colisions(self,grid_cells):
        from map import TILE
        ix = int((self.x) // TILE)
        iy = int((self.y) // TILE)
        cell = grid_cells[ix][iy]

        expected_x = self.x + self.velx
        expected_y = self.y + self.vely

        if expected_y  - self.size/2 - (iy * TILE) < 0:
            if cell.walls['top']:
                self.vely = self.dec
        if expected_y  + self.size/2 - (iy * TILE) > TILE:
            if cell.walls['bot']:
                self.vely = - self.dec
        if expected_x  - self.size/2 - (ix * TILE) < 0:
            if cell.walls['left']:
                self.velx = self.dec
        if expected_x  + self.size/2 - (ix * TILE) > TILE:
            if cell.walls['right']:
                self.velx = -self.dec


        self.x = self.x + self.velx
        self.y = self.y + self.vely

    def update(self,grid_cells):
        if self.move_left and self.velx > - self.max_speed:
            self.velx -= self.acc
        if self.move_right and self.velx < self.max_speed:
            self.velx += self.acc
        if self.move_up and self.vely > - self.max_speed:
            self.vely -= self.acc
        if self.move_down and self.vely < self.max_speed:
            self.vely += self.acc
            
        if self.velx > 0 : self.velx -= self.dec
        if self.velx < 0 : self.velx += self.dec
        if self.vely > 0 : self.vely -= self.dec
        if self.vely < 0 : self.vely += self.dec

        self.check_colisions(grid_cells)

    def get_pos(self):
        return self.x,self.y

    def propagate_sound(self,grid_cells,mics,t):
        self.noise.propagate_sound(self.get_pos(),grid_cells,mics,t)
        for mic in mics:
            cell = grid_cells[mic.cell_idx[0]][mic.cell_idx[1]]
            distx = abs(mic.x-cell.pos[0])
            disty = abs(mic.y-cell.pos[1])
            dist = distx + disty + cell.dist
            mic.record(self.noise.get_apparent_sound(t,dist),t)
        """
        if cell.mic is not None:
            mic = mics[cell.mic]
            diffx = abs(mic.x-x)
            diffy = abs(mic.y-y)
            dist += diffx + diffy
            mic.record(self.get_apparent_sound(t,dist),t)
            return
            """
