import pygame
import random
from map import *
from player import *
from noise import *
from MSE_controller import *

def MSE(method = 'Fuzzy'):
    FPS = 30

    pygame.init()

    grid_cells, mics = get_map(43)

    if method == 'Linear':
        controller = MSE_linear(grid_cells,mics)
    else:
        controller = MSE_Fuzzy(grid_cells,mics)

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

    
    # Test performance
    test = 0
    when = 6
    all_errors = []
    total_error = 0

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

        # Estimate position & measure performance
        est_pos = controller.get_pos_prediction()
        if test == when:
            test = 0
            real_pos = player.get_pos()
            error = np.abs(real_pos[0] - est_pos[0]) + np.abs(real_pos[1] - est_pos[1])
            all_errors = np.append(all_errors, error)
            total_error += error


        
        controller.draw(sc)


        pygame.display.flip()


        test += 1
        t += 1/FPS
        clock.tick(FPS)

    for i,mic in enumerate(mics):
        mic.save(i)

    # Performance stats
    mean_error = total_error/np.size(all_errors)
    print("Max error: ", np.max(all_errors))
    print("Min error: ", np.min(all_errors))
    print("Mean error: ", mean_error)
    print("Distance: ", 102*TILE)
    print("Course time (s): ", (np.size(all_errors)*when)/FPS)
    print("Max speed: ", player.max_speed)
    print("Sampling frequency (/s): ", FPS/when)