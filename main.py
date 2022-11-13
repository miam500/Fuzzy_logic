import pygame
import random
from map import *
from player import *
from noise import *
from fuzzyLogic import *

FPS = 30

pygame.init()

grid_cells, mics = get_map(43)

controller = MSE_linear(grid_cells,mics)
sc = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()

player = Player(MIDDLE)
player.add_noise(Constant_Noise(40))

sc.fill(COLOR_UNVISITED)

for row in grid_cells:
    for cell in row :
        cell.draw(sc)

t = 0
loop = True
while loop:

    for row in grid_cells:
        for cell in row :
            cell.draw(sc)
            cell.dist = 9999

    for mic in mics:
        mic.draw(sc)
        
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            loop = False
            break
        if event.type == pygame.KEYDOWN:
            player.move(event.key)
        if event.type == pygame.KEYUP:
            player.stop_moving(event.key)
            
    player.update(grid_cells)
    player.draw(sc)
    

    player.propagate_sound(grid_cells,mics,t)

    
    controller.draw(sc)


    pygame.display.flip()

    t += 1/FPS
    clock.tick(FPS)

for i,mic in enumerate(mics):
    mic.save(i)