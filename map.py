import random
import pygame
from matplotlib import pyplot as plt

MAP_FOLDER = 'maps\\'
RESOLUTION = WIDTH, HEIGHT = 900, 900
TILE = 100
MIDDLE = WIDTH/2,HEIGHT/2
WALL_BUST = 10
cols, rows = WIDTH // TILE, HEIGHT // TILE
MIC_DIST_FROM_WALL = 5

COLOR_WALL = pygame.Color('darkorange')
COLOR_MIC = pygame.Color('green')
COLOR_UNVISITED = pygame.Color('darkslategray')
COLOR_VISITED = pygame.Color('black')
COLOR_CURRENT = pygame.Color('yellow')
COLOR_PREDICTION = pygame.Color('blue')

def get_map(seed, nmb_mic=3):
    from os.path import exists
    from os import getcwd
    
    filename = getcwd() +'\\' + MAP_FOLDER +'grid_'+str(seed)+'.txt'
    pygame.display.set_caption(filename)
    if exists(filename):
        grid_cells =load_map(filename)
    else:
        grid_cells =  generate_map(seed)
    
    mics = place_mics(grid_cells,nmb_mic,seed)

    return grid_cells, mics

def load_map(filename):
    grid_cells = []
    with open(filename,'r') as f:
        for line in f:
            row = []
            cells = line.rstrip().split(';')
            for cell in cells:
                if cell:
                    x,y,walls = cell.split(',')
                    x,y = int(x.strip()),int(y.strip())
                    walls = [bool(int(walls.strip()[i])) for i in range(4)]
                    row.append(Cell(x,y,walls))
            grid_cells.append(row)
    return grid_cells

def generate_map(seed):
    random.seed(seed)

    sc = pygame.display.set_mode(RESOLUTION)
    clock = pygame.time.Clock()
    grid_cells = [[Cell(col,row) for row in range(rows)] for col in range(cols)]
    num_unvisited_cells = cols * rows
    current_cell = grid_cells[0][0]

    sc.fill(COLOR_UNVISITED)

    for row in grid_cells:
        for cell in row :
            cell.draw(sc)

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        
        if not current_cell.visited:
            num_unvisited_cells -= 1

        current_cell.visited = True

        if num_unvisited_cells < 1:
            break


        next_cell = current_cell.get_next_cell(grid_cells)
        if next_cell:
            dx = next_cell.x - current_cell.x
            dy = next_cell.y - current_cell.y
            current_cell.remove_wall(dx,dy)
            next_cell.remove_wall(-dx,-dy)
            current_cell.draw(sc)
            next_cell.draw_current_cell(sc)
            current_cell = next_cell

        
        pygame.display.flip()
        clock.tick(100)
    
    save_map(grid_cells,seed)
    return grid_cells

def save_map(grid,seed):

    with open(MAP_FOLDER+'grid_'+str(seed)+'.txt','w') as f:
        for row in grid:
            for cell in row:
                f.write(str(cell))
                f.write(';')
            f.write('\n')

def place_mics(grid_cells,n_mics,seed):
    random.seed(seed)
    n_rows = len(grid_cells)
    n_cols = len(grid_cells[0])

    mics = []
    for mic in range(n_mics):
        x = random.randrange(0,n_rows)
        y = random.randrange(0,n_cols)
        cell_with_mic = grid_cells[x][y].place_mic(grid_cells)
        mics.append(Mic(cell_with_mic))
        cell_with_mic.mic = mic

    return mics


class Cell:
    def __init__(self, x, y, walls = None):
        self.x, self.y = x, y
        if walls:
            self.walls = {'top': walls[0], 'bot': walls[1], 'left': walls[2],'right': walls[3] }
            self.visited = True
        else:
            self.walls = {'top': True, 'bot': True, 'left': True,'right': True }
            self.visited = False
        self.mic = None
        self.dist = 999
        self.pos = (0,0)


    def draw(self,sc, color=None):
        x, y = self.x * TILE, self.y * TILE
        if self.visited & (color is None):
            pygame.draw.rect(sc, COLOR_VISITED, (x, y, TILE, TILE))
        elif color is not None:
            pygame.draw.rect(sc, color, (x, y, TILE, TILE))
            
        if self.walls['top']:
            pygame.draw.line(sc, COLOR_WALL, (x, y), (x + TILE, y), 1)
        if self.walls['bot']:
            pygame.draw.line(sc, COLOR_WALL, (x, y + TILE-1), (x + TILE, y + TILE-1), 1)
        if self.walls['left']:
            pygame.draw.line(sc, COLOR_WALL, (x, y), (x, y + TILE), 1)
        if self.walls['right']:
            pygame.draw.line(sc, COLOR_WALL, (x + TILE-1, y), (x + TILE-1, y + TILE), 1)
    
    def draw_current_cell(self,sc):
        x, y = self.x * TILE, self.y * TILE
        pygame.draw.rect(sc, COLOR_CURRENT, (x+2, y+2, TILE-2, TILE-2))

    def get_neighbors(self,grid_cells):
        valid_neighbors = []
        x,y = self.x,self.y
        if x - 1 >= 0:
            if not grid_cells[x-1][y].visited:
                valid_neighbors.append(grid_cells[x-1][y])
        if x + 1 < cols:
            if not grid_cells[x+1][y].visited:
                valid_neighbors.append(grid_cells[x+1][y])
        if y - 1 >= 0:
            if not grid_cells[x][y-1].visited:
                valid_neighbors.append(grid_cells[x][y-1])
        if y + 1 < rows:
            if not grid_cells[x][y+1].visited:
                valid_neighbors.append(grid_cells[x][y+1])

        if not valid_neighbors:
            if y + 1 < rows:
                if not grid_cells[x][y+1].walls['top'] or random.uniform(1,100) < WALL_BUST:
                    valid_neighbors.append(grid_cells[x][y+1])
            if y - 1 >= 0:
                if not grid_cells[x][y-1].walls['bot'] or random.uniform(1,100) < WALL_BUST:
                    valid_neighbors.append(grid_cells[x][y-1])
            if x + 1 < cols:
                if not grid_cells[x+1][y].walls['left'] or random.uniform(1,100) < WALL_BUST:
                    valid_neighbors.append(grid_cells[x+1][y])
            if x - 1 >= 0:
                if not grid_cells[x-1][y].walls['right'] or random.uniform(1,100) < WALL_BUST:
                    valid_neighbors.append(grid_cells[x-1][y])

        return valid_neighbors
    
    def get_next_cell(self,grid_cells):
        return random.choice(self.get_neighbors(grid_cells))
    
    def place_mic(self,grid_cells):
        valid_walls = [name for name in ['top','bot','left','right'] if self.walls[name]]
        if self.mic or len(valid_walls) == 0:
            cell_with_mic = self.get_next_cell(grid_cells).place_mic(grid_cells)
        else:
            self.mic = random.choice(valid_walls)
            cell_with_mic = self
        
        return cell_with_mic

    def remove_wall(self, dx,dy):
        if dy == 1:
            self.walls['bot'] = False
        if dy == -1:
            self.walls['top'] = False
        if dx == -1:
            self.walls['left'] = False
        if dx == 1:
            self.walls['right'] = False
    
    def __repr__(self):
        walls = ''
        for edge in ['top','bot','left','right']:
            walls += str(int(self.walls[edge])) 
        return str(self.x) + ', '+ str(self.y)+ ', '+ walls
   
class Mic:
    def __init__(self,cell):
        self.x = cell.x * TILE
        self.y = cell.y * TILE
        self.cell_idx = (cell.x,cell.y)

        if cell.mic == 'top':
            self.x += TILE // 2
            self.y += MIC_DIST_FROM_WALL
        if cell.mic == 'bot':
            self.x += TILE // 2
            self.y += TILE - MIC_DIST_FROM_WALL
        if cell.mic == 'left':
            self.x += MIC_DIST_FROM_WALL
            self.y += TILE // 2
        if cell.mic == 'right':
            self.x += TILE - MIC_DIST_FROM_WALL
            self.y += TILE // 2


        self.data = []
        self.t = []

    def draw(self,sc,sound_path = False):
        pygame.draw.rect(sc, COLOR_MIC, (self.x-MIC_DIST_FROM_WALL, self.y-MIC_DIST_FROM_WALL, MIC_DIST_FROM_WALL*2, MIC_DIST_FROM_WALL*2))

    def record(self,amp,t):
        self.t.append(t)
        self.data.append(amp)

    def save(self,num):
        fig = plt.figure()
        plt.plot(self.t,self.data)
        fig.savefig('Mic_'+str(num)+'.png')

