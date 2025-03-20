import pygame
import numpy as np
import torch

from config import *
from tetromino import Tetromino


class TetrisGame:


    #############################
    #           Init            #
    #############################

    def __init__(self, ui=True):
        if ui :
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Tetris")
            self.font = pygame.font.SysFont('Arial', 24)
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.current_piece = Tetromino(GRID_WIDTH//2 - 2,0)
        self.next_piece = Tetromino(0,0)
        self.score = 0
        self.level = 0
        self.lines = 0
        self.fall_speed = START_SPEED
        self.last_fall = pygame.time.get_ticks()
        self.running = True
        self.full_lines = []
        self.reward = 0
        self.empty_lines = 20
        self.holes = 0

        return self.state_data()
    


    #############################
    #           Ctrl            #
    #############################
    
    def move(self, dx=0, dy=0):
        if not self.current_piece.collision(dx, dy, self.grid):
            self.current_piece.x += dx
            self.current_piece.y += dy
    
    def hard_drop(self):
        while not self.current_piece.collision(0, 1, self.grid):
            self.current_piece.y += 1
        now = pygame.time.get_ticks()
        self.lock_piece()
        self.last_fall = now

    def rotate_piece(self):
        self.current_piece.rotate(self.grid)



    #############################
    #           Update          #
    #############################

    def update(self, now):
        if not self.current_piece.collision(0, 1, self.grid):
            self.current_piece.y += 1
            self.last_fall = now
        else:
            self.lock_piece()
            self.last_fall = now
    
    def lock_piece(self):
        for y, row in enumerate(self.current_piece.get_shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[self.current_piece.y + y][self.current_piece.x + x] = self.current_piece.color
        self.clear_lines()
        self.current_piece = self.next_piece
        self.next_piece = Tetromino(0,0)
        self.current_piece.x = GRID_WIDTH//2 - 2
        self.current_piece.y = 0
        if self.current_piece.collision(0, 0, self.grid):
            self.running = False
    
    def clear_lines(self):
        self.full_lines = []
        for i, row in enumerate(self.grid):
            if 0 not in row:
                self.full_lines.append(i)
        
        for line in self.full_lines:
            self.grid = np.delete(self.grid, line, 0)
            self.grid = np.insert(self.grid, 0, 0, axis=0)
        
        if self.full_lines:
            self.lines += len(self.full_lines)
            self.score += 100 * (2 ** len(self.full_lines) - 1)
            self.level = 0 + self.lines // 10
            self.fall_speed = max(50, START_SPEED - 75 * self.level)

    #######################################################################################
    ###############                            Fct UI                      ################
    #######################################################################################

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x]:
                    pygame.draw.rect(self.screen, COLORS[self.grid[y][x]],
                                   (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE-1, BLOCK_SIZE-1))
    
    def draw_piece(self, piece, offset_x=0, offset_y=0):
        for y, row in enumerate(piece.get_shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, COLORS[piece.color],
                                    ((piece.x + x + offset_x)*BLOCK_SIZE,
                                     (piece.y + y + offset_y)*BLOCK_SIZE,
                                     BLOCK_SIZE-1, BLOCK_SIZE-1))
    
    def draw_ui(self):
        # Score
        text = self.font.render(f'Score: {self.score}', True, (255,255,255))
        self.screen.blit(text, (BLOCK_SIZE*GRID_WIDTH + 10, 20))
        text = self.font.render(f'Lvl: {self.level}', True, (255,255,255))
        self.screen.blit(text, (BLOCK_SIZE*GRID_WIDTH + 10, 60))
        
        # Prochaine pièce
        text = self.font.render('Next:', True, (255,255,255))
        self.screen.blit(text, (BLOCK_SIZE*GRID_WIDTH + 10, 100))
        self.draw_piece(self.next_piece, GRID_WIDTH + 2, 5)

        pygame.draw.line(self.screen, (255, 255, 255), (BLOCK_SIZE*GRID_WIDTH, 0), (BLOCK_SIZE*GRID_WIDTH, self.screen.get_height()), 2)
    








    #######################################################################################
    ###############               Fct pour jeux classique                  ################
    #######################################################################################

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.screen.fill((0,0,0))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move(-1)
                    if event.key == pygame.K_RIGHT:
                        self.move(1)
                    if event.key == pygame.K_DOWN:
                        self.move(0,1)
                    if event.key == pygame.K_UP:
                        self.rotate_piece()
                    if event.key == pygame.K_SPACE:
                        self.hard_drop()
            
            now = pygame.time.get_ticks()
            if now - self.last_fall > self.fall_speed:
                self.update(now)
            self.draw_grid()
            self.draw_piece(self.current_piece)
            self.draw_ui()
            pygame.display.update()
        
        pygame.quit()










    #######################################################################################
    ###############                Fct d'etat pour IA                      ################
    #######################################################################################

    #############################
    #        Fct Reward         #
    #############################

    def update_reward(self):
        self.remplir_lignes()
        self.maximiser_lignes_vides()
        self.minimiser_trous()
        self.afficher_stats()

    def remplir_lignes(self):
        if self.full_lines:
            self.reward += 100 * (2 ** len(self.full_lines) - 1)
        else :
            self.reward -= 5

    def maximiser_lignes_vides(self):
        empty_lines = 0
        for row in self.grid: # Parcours les lignes
            if 1 not in row: # Si aucune case pleine
                empty_lines += 1 # Alors incrément du nombre de lignes vides
        #if empty_lines > self.empty_lines: # Si nbr de ligne vide supérieur au nbr de lignes vides avant la pièce alors pénalité
        self.reward += 10 * (empty_lines - self.empty_lines) # Valeur arbitraire
        self.empty_lines = empty_lines # Save du nombre de ligne vides actuelles

    
    def minimiser_trous(self):
        holes = 0
        for i, row in enumerate(self.grid): # Parcours les lignes
            for j in row: # Parcours les colonnes de la ligne i
                if not j: # Si case vide
                    for k in range(i-1, -1, -1): # Parcours les cases supérieur de la colonne j depuis la ligne i
                        if self.grid[k][j] == 1: # Si case pleine
                            holes += 1 # Alors incrément nbr trous
                            break
        #if holes > self.holes:
        self.reward -= 30 * (holes - self.holes) # Valeur arbitraire
        self.holes = holes

    def afficher_stats(self):
        print("Lignes vides : ", self.empty_lines)
        print("Trous : ", self.holes)
        print("Récompenses : ", self.reward)

        
    #############################
    #           Fct Step        #
    #############################

    def step(self, action, ui=False):
        self.reward = 0
        self.clock.tick(FPS)
        if ui :
            self.screen.fill((0,0,0))

        if action == 0:
            self.move(-1)
        if action == 1:
            self.move(1)
        if action == 2:
            self.move(0,1)
        if action == 3:
            self.rotate_piece()
        #if action == 4:
            #self.hard_drop()
    
        now = pygame.time.get_ticks()
        if now - self.last_fall > self.fall_speed:
            self.update(now)
            self.update_reward()
        if ui :
            self.draw_grid()
            self.draw_piece(self.current_piece)
            self.draw_ui()
            pygame.display.update()

        return self.state_data()



    #############################
    #         State Data        #
    #############################

    def state_data(self):
        shape_data_np = np.array(self.current_piece.shape_data)  # Convertit en numpy array
        h, w = shape_data_np.shape # Récupère la taille actuelle
        normalized_shape_data = np.zeros((4, 4), dtype=np.float32)
        normalized_shape_data[:h, :w] = shape_data_np

        shape_data_np = np.array(self.next_piece.shape_data)  # Convertit en numpy array
        h, w = shape_data_np.shape  # Récupère la taille actuelle
        normalized_next_shape_data = np.zeros((4, 4), dtype=np.float32)
        normalized_next_shape_data[:h, :w] = shape_data_np


        grid_tensor = torch.tensor(self.grid, dtype=torch.float32).flatten()  # (200,)
        tetromino_tensor = torch.tensor(normalized_shape_data, dtype=torch.float32).flatten()  # (16,)
        position_tensor = torch.tensor([self.current_piece.x, self.current_piece.y], dtype=torch.float32)  # (2,)
        next_tetromino_tensor = torch.tensor(normalized_next_shape_data, dtype=torch.float32).flatten()  # (16,)

        # Concaténation de toutes les entrées
        state_tensor = torch.cat([grid_tensor, tetromino_tensor, position_tensor, next_tetromino_tensor])

        return state_tensor, self.reward, not self.running
    




