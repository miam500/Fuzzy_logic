import pygame
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from map import *
import random
from player import *
from noise import *

def fuzzy_technique_2():
    FPS = 30

    pygame.init()

    grid_cells, mics = get_map(43)

    sc = pygame.display.set_mode(RESOLUTION)
    clock = pygame.time.Clock()
    ancien_pos=[4,4]

    player = Player([ancien_pos[0]*TILE,ancien_pos[1]*TILE])
    player.add_noise(Constant_Noise(40))

    sc.fill(COLOR_UNVISITED)

    for row in grid_cells:
        for cell in row :
            cell.draw(sc)
    controller =createFuzzyController(grid_cells,sc, mics,player)

    print(controller.sim.input)
    t = 0
    loop = True
    player.propagate_sound(grid_cells,mics,t)
    resultat_micro=[]
    ancien_resultat_micro=[]  
    for k in range(len(mics)):
        print(len(mics[k].data))
        ancien_resultat_micro.append(mics[k].data[-1])
        resultat_micro.append(mics[k].data[-1])
    donnee=[0 for i in range(len(mics))]

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
        
        print(ancien_resultat_micro)
        for k in range(len(mics)):
            resultat_micro[k]=mics[k].data[-1]
       
        ancien_pos=objective(controller,resultat_micro)
        rect = pygame.Rect(int(ancien_pos[0]),int(ancien_pos[1]) ,TILE,TILE)
        pygame.draw.rect(sc,(120,250,60),rect)


        pygame.display.flip()

        t += 1/FPS
        clock.tick(FPS)

    for i,mic in enumerate(mics):
        mic.save(i)
class createFuzzyController():
    def __init__(self, grid_cells,sc, mics,player,t=0, sampling_grid=2, membership_fcns=5,):
        self.n_membership = membership_fcns
        self.grid = grid_cells
        self.mic = mics
        self.size = 30
        self.color = pygame.Color('red')
        self.sampling_grid = sampling_grid
        nb_case_largeur=int(HEIGHT/TILE)
        nb_case_longueur=int(WIDTH/TILE)
        self.tableau_regle=self.etablissement_des_regles(player,grid_cells,t,sc)
        self.precision=5
        
        self.antecedent=[]
        for i in range(len(self.mic)):
            self.antecedent.append(ctrl.Antecedent(np.arange(self.calcul_min()-10, self.calcul_max()+10,0.001), 'mic'+str(i)))
        
        
        
        x_out = ctrl.Consequent(np.arange(0, WIDTH,1), 'x_out',defuzzify_method='centroid')
        y_out = ctrl.Consequent(np.arange(0, HEIGHT,1), 'y_out',defuzzify_method='centroid')
        
        fonction_appartenance_micro = [str(i) for i in range(self.precision)]
        
        intervalle=(self.calcul_max()-self.calcul_min())/(self.precision-1)
        minimun=self.calcul_min()
        for i in range(len(self.antecedent)):
            self.antecedent[i][fonction_appartenance_micro[0]]=fuzz.trapmf(self.antecedent[i].universe, [minimun-10, minimun-10, minimun,minimun+intervalle])
            for j in range(1,self.precision-1):
                self.antecedent[i][fonction_appartenance_micro[j]]=fuzz.trimf(self.antecedent[i].universe, [minimun+(j-1)*intervalle, minimun+j*intervalle,minimun+(j+1)*intervalle])
            self.antecedent[i][fonction_appartenance_micro[self.precision-1]]=fuzz.trapmf(self.antecedent[i].universe, [minimun+(self.precision-2)*intervalle, self.calcul_max(), self.calcul_max()+10,self.calcul_max()+10])
        
        for i in range(nb_case_longueur):
            x_out[str(i)]=fuzz.trimf(x_out.universe,[(2*(i-1)+1)*TILE/2, (2*i+1)*TILE/2,(2*(i+1)+1)*TILE/2])
            #x_out[str(i)]=fuzz.trimf(x_out.universe,[i-1, i,i+1])
            
            
        
        for i in range(nb_case_largeur):
            y_out[str(i)]=fuzz.trimf(y_out.universe,[(2*(i-1)+1)*TILE/2, (2*i+1)*TILE/2,(2*(i+1)+1)*TILE/2])
            #y_out[str(i)]=fuzz.trimf(y_out.universe,[i-1, i,i+1])
            

        
        self.conversion_valeur_fonction_appartenance(fonction_appartenance_micro)
        
        rules = []
        
        for j in range(len(self.tableau_regle)):
            for i in range(len(self.tableau_regle[j])):
                
                rules.append(ctrl.Rule(antecedent=(self.antecedent[0][self.tableau_regle[j][i][0]] & self.antecedent[1][self.tableau_regle[j][i][1]] & self.antecedent[2][self.tableau_regle[j][i][2]] ), consequent=(x_out[str(int(i/2))])))
                
                rules.append(ctrl.Rule(antecedent=(self.antecedent[0][self.tableau_regle[j][i][0]] & self.antecedent[1][self.tableau_regle[j][i][1]] & self.antecedent[2][self.tableau_regle[j][i][2]] ), consequent=(y_out[str(int(j/2))])))
                
        for rule in rules:
            rule.and_func = np.multiply
            rule.or_func = np.max
    
        system = ctrl.ControlSystem(rules=rules)
        
        self.sim = ctrl.ControlSystemSimulation(system)
        # for rule in sim.ctrl.rules:
        #     print(rule)
        # self.sim = sim
        # for var in self.sim.ctrl.fuzzy_variables:
        #     var.view()
        # plt.show()

    def get_input(self):
        return self.mic.data
    
    def etablissement_des_regles(self,player,grid_cells,t,sc):
        nb_case_largeur=int(HEIGHT/TILE)
        nb_case_longueur=int(WIDTH/TILE)
        tableau_regle=[]
        for j in range(nb_case_largeur):
            ligne_tableau=[]
            for i in range(nb_case_longueur):
                resultat_micro=[]
                player.x=(i+1)*TILE/2
                player.y=(j+1)*TILE/2
                player.draw(sc)
                player.propagate_sound(grid_cells,self.mic,t)
                for k in range(len(self.mic)):
                    resultat_micro.append(self.mic[k].data[-1])
                ligne_tableau.append(resultat_micro)
            tableau_regle.append(ligne_tableau)
        
        
        
        return tableau_regle
    def calcul_min(self):
        min=self.tableau_regle[0][0][0]
        for j in range(len(self.tableau_regle)):
            for i in range(len(self.tableau_regle[j])):
                for k in range(len(self.mic)):
                    if min>self.tableau_regle[j][i][k]:
                        min=self.tableau_regle[j][i][k]
        return min
    def conversion_valeur_fonction_appartenance(self,fonction_appartenance_input):
        intervalle=(self.calcul_max()-self.calcul_min())/((self.precision-1)*2)
        min=self.calcul_min()
        for j in range(len(self.tableau_regle)):
            for i in range(len(self.tableau_regle[j])):
                for k in range(len(self.mic)):
                    for l in range(self.precision):
                        if self.tableau_regle[j][i][k]<min+(2*l+1)*intervalle:
                            self.tableau_regle[j][i][k]=fonction_appartenance_input[l]
                            break
                    
                    
    def calcul_max(self):
        max=self.tableau_regle[0][0][0]
        for j in range(len(self.tableau_regle)):
            for i in range(len(self.tableau_regle[j])):
                for k in range(len(self.mic)):
                    if max<self.tableau_regle[j][i][k]:
                        max=self.tableau_regle[j][i][k]
        return max
def objective(fuzz_ctrl,data_micro):
    
    
    for i in range(len(fuzz_ctrl.mic)):
        
        fuzz_ctrl.sim.input['mic'+str(i)] = data_micro[i]
    
    
    fuzz_ctrl.sim.compute()
    return fuzz_ctrl.sim.output['x_out'],fuzz_ctrl.sim.output['y_out']