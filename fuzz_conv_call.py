from player import *
from noise import *
from fuzzyLogic_conventional import *


def test_conventinnal():
    FPS = 30
    NMB_MICS = 6
    
    pygame.init()
    
    grid_cells, mics = get_map(seed=43, nmb_mic=NMB_MICS)
    
    ctrls = {}
    for i, mic in enumerate(mics):
        key = f"mic_{i}"
        ctrls[key] = createFuzzyController(grid_cells=grid_cells, mic=mic)
        for var in ctrls[key].sim.ctrl.fuzzy_variables:
            var.view()
    
    plt.show()
    
    sc = pygame.display.set_mode(RESOLUTION)
    clock = pygame.time.Clock()
    
    player = Player(MIDDLE)
    player.add_noise(Constant_Noise(40))
    
    sc.fill(COLOR_UNVISITED)
    
    for row in grid_cells:
        for cell in row:
            cell.draw(sc)
    
    
    t = 0
    loop = True
    while loop:
    
        for row in grid_cells:
            for cell in row:
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
    
        player.propagate_sound(grid_cells, mics, t)
        L = []
        for i, mic in enumerate(mics):
            ctrls[f'mic_{i}'].sim.input['mic'] = mic.data[-1]
            ctrls[f'mic_{i}'].sim.input['x'] = mic.x
            ctrls[f'mic_{i}'].sim.input['y'] = mic.y
            ctrls[f'mic_{i}'].sim.compute()
            L.append((ctrls[f'mic_{i}'].sim.output['x_out'], ctrls[f'mic_{i}'].sim.output['y_out']))
    
        x_final = 0
        y_final = 0
        for i in range(len(L)-1):
            x_final += (L[i][0] - x_final)/2
            y_final += (L[i][1] - y_final)/2
    
    
        pygame.draw.circle(sc, color='red', center=(abs(x_final), abs(y_final)), radius=15)
    
        pygame.display.flip()
    
        t += 1 / FPS
        clock.tick(FPS)
