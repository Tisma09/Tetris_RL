
# Configuration du jeu
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
WIDTH = BLOCK_SIZE * (GRID_WIDTH + 6)  # Espace pour l'UI
HEIGHT = BLOCK_SIZE * GRID_HEIGHT
FPS = 60
START_SPEED = 1000

# Couleurs
COLORS = [
    (0,   0,   0),    # Noir - fond
    (255, 0,   0),    # Rouge
    (0,   150, 0),    # Vert
    (0,   0,   255),  # Bleu
    (255, 120, 0),    # Orange
    (255, 255, 0),    # Jaune
    (180, 0,   255),  # Violet
    (0,   220, 220)   # Cyan
]

# Formes et rotations des Tetrominos
SHAPES = [
    [[1, 1, 1, 1]],                          # I
    [[2, 2], [2, 2]],                        # O
    [[0, 3, 0], [3, 3, 3]],                  # T
    [[0, 4, 4], [4, 4, 0]],                  # S
    [[5, 5, 0], [0, 5, 5]],                  # Z
    [[6, 0, 0], [6, 6, 6]],                  # J
    [[0, 0, 7], [7, 7, 7]]                   # L
]