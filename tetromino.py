import random

from config import *


class Tetromino:
    def __init__(self, x, y):
        self.shape_data = random.choice(SHAPES)
        self.color = SHAPES.index(self.shape_data) + 1
        self.x = x
        self.y = y

    @property
    def get_shape(self):
        return self.shape_data
        
    def rotate(self, grid, nb=1):
        old_shape = self.shape_data
        for i in range(nb) :
            self.shape_data = [list(row) for row in zip(*self.shape_data[::-1])]
        if self.collision(0, 0, grid):
            self.shape_data = old_shape

    
    def collision(self, dx, dy, grid):
        for y, row in enumerate(self.get_shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.x + x + dx
                    new_y = self.y + y + dy
                    if new_x < 0 or new_x >= GRID_WIDTH:
                        return True
                    if new_y >= GRID_HEIGHT:
                        return True
                    if new_y >= 0 and grid[new_y][new_x]:
                        return True
        return False
