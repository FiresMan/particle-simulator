import pygame
import math
import numpy as np

pygame.init()

TIMESTEP = 0.1
UNIT = 10
SMOTH_COEF_A = 10
SMOTH_COEF_B = 1
RADIUS = 2
WIDTH = 800
HEIGHT = 800
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
FONT = pygame.font.SysFont('Times New Roman', 16)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Particles!')


class Particle:
    atr_table = [[1, 1,  1],
                 [1, 1,  1],
                 [1, 1,  1]]
    rep_table = [[16, 16,  16],
                 [16, 16,  16],
                 [16, 16,  16]]    
    color_table = [GREEN, RED, BLUE]
    
    def __init__(self, p_type, x_vec, v_vec = np.array([0, 0]), a_vec = np.array([0, 0])):
        self.type = p_type
        self.x_vec = x_vec
        self.v_vec = v_vec
        self.a_vec = a_vec
    
    def return_vec(self):
        return np.concatenate([self.x_vec, self.v_vec])

class Simulation:
    def __init__(self, p_list):
        self.n_bodies = len(p_list)
        self.n_dim = 4
        self.quant_vec = np.concatenate([i.return_vec() for i in p_list])
        self.type_vec = np.array([i.type for i in p_list])
    
    
    def set_diff_eq(self,calc_diff_eqs,**kwargs):
        self.diff_eq_kwargs = kwargs
        self.calc_diff_eqs = calc_diff_eqs
    
    def add_particle(self, particle):
        self.n_bodies += 1
        self.quant_vec = np.append(self.quant_vec, particle.return_vec())
        self.type_vec = np.append(self.type_vec, particle.type)
        
    
    def rk4(self, t, dt):
        k1 = dt * self.calc_diff_eqs(t, self.quant_vec, self.type_vec, **self.diff_eq_kwargs) 
        k2 = dt * self.calc_diff_eqs(t + 0.5*dt, self.quant_vec + 0.5*k1, self.type_vec, **self.diff_eq_kwargs)
        k3 = dt * self.calc_diff_eqs(t + 0.5*dt, self.quant_vec + 0.5*k2, self.type_vec, **self.diff_eq_kwargs)
        k4 = dt * self.calc_diff_eqs(t + dt, self.quant_vec + k3, self.type_vec, **self.diff_eq_kwargs)
            
        y_new = self.quant_vec + (k1+ 2*k2 + 2*k3 + k4) / 6.0
        return y_new
    
    def update(self, t, dt):
        self.quant_vec = self.rk4(t, dt)
        for i in range(self.n_bodies):
            offset = i * 4
            if abs(self.quant_vec[offset]) > WIDTH / (2 * UNIT):
                if self.quant_vec[offset] > 0:
                    self.quant_vec[offset] = WIDTH / (2 * UNIT)
                else:
                    self.quant_vec[offset] = -WIDTH / (2 * UNIT)
                self.quant_vec[offset + 2] *= -1
            if abs(self.quant_vec[offset + 1]) > HEIGHT / (2 * UNIT):
                if self.quant_vec[offset + 1] > 0:
                    self.quant_vec[offset + 1] = HEIGHT / (2 * UNIT)
                else:
                    self.quant_vec[offset + 1] = -HEIGHT / (2 * UNIT)
                self.quant_vec[offset + 3] *= -1
        
    
    def run(self, t, dt, d_0 = 0):
        history = []
        q_vec = self.quant_vec
        history.append(q_vec)
        for step in range(t // dt):
            d_0 += dt
            q_vec = self.rk4(d_0, dt)
            history.append(q_vec)
        return history
    
    def draw(self):
        for i in range(self.n_bodies):
            offset = i * 4
            x = self.quant_vec[offset] * UNIT + WIDTH / 2
            y = self.quant_vec[offset + 1] * UNIT + HEIGHT / 2            
            pygame.draw.circle(screen, Particle.color_table[self.type_vec[i] - 1], (x, y), RADIUS)
            
    
def nbody_solve(t, y, types):
    N_bodies = int(len(y) / 4)
    solved_vector = np.zeros(y.size)
    for i in range(N_bodies):
        ioffset = i * 4 
        for j in range(N_bodies):
            joffset = j*4
            solved_vector[ioffset] = y[ioffset+2]
            solved_vector[ioffset+1] = y[ioffset+3]
            if i != j:
                dx = y[joffset] - y[ioffset]
                dy = y[ioffset+1] - y[joffset+1]
                r = (dx**2+dy**2)**0.5
                if r != 0:
                    ax = ((Particle.atr_table[types[i] - 1][types[j] - 1] / r**3) * dx  - (Particle.rep_table[types[i] - 1][types[j] - 1] / r**7) * dx) * (2 / (1 + math.exp(-SMOTH_COEF_A * (r / SMOTH_COEF_B)**2)) - 1)
                    ay = ((Particle.atr_table[types[i] - 1][types[j] - 1] / r**3) * dy - (Particle.rep_table[types[i] - 1][types[j] - 1] / r**7) * dy) * (2 / (1 + math.exp(-SMOTH_COEF_A * (r / SMOTH_COEF_B)**2)) - 1)
                else:
                    ax = ay = 0
                solved_vector[ioffset+2] += ax
                solved_vector[ioffset+3] += ay
    return solved_vector

def main():
    run = True
    t_run = False
    clock = pygame.time.Clock()
    t = 0
    
    p_list = [Particle(1, np.array([0, 0])), Particle(2, np.array([2, 0]))]
    
    simulation = Simulation(p_list)
    simulation.set_diff_eq(nbody_solve)
    
    while run:
        clock.tick(60)
        screen.fill((0, 0, 0))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                t_run = not t_run
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and event.pos[0] >= 0 and event.pos[0] <= WIDTH and event.pos[1] >= 0 and event.pos[1] <= HEIGHT:
                if pygame.key.get_pressed()[pygame.K_1]:
                    simulation.add_particle(Particle(1, np.array([(event.pos[0] - WIDTH / 2)/ UNIT, (event.pos[1] - HEIGHT / 2)/ UNIT])))
                elif pygame.key.get_pressed()[pygame.K_2]:
                    simulation.add_particle(Particle(2, np.array([(event.pos[0] - WIDTH / 2)/ UNIT, (event.pos[1] - HEIGHT / 2)/ UNIT])))
                elif pygame.key.get_pressed()[pygame.K_3]:
                    simulation.add_particle(Particle(3, np.array([(event.pos[0] - WIDTH / 2)/ UNIT, (event.pos[1] - HEIGHT / 2)/ UNIT])))
                
        
        if t_run:
            simulation.update(t, TIMESTEP)
            t += TIMESTEP
            
        simulation.draw()
        pygame.display.update()
    
    
    pygame.quit()


main()
            
