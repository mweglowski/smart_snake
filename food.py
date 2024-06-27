import pygame
import random

BLOCK_SIZE = 50
SCREEN_WIDTH = BLOCK_SIZE * 10
SCREEN_HEIGHT = BLOCK_SIZE * 10

class Food:
    def __init__(self):
        # Place food, so that it is located exactly within random cell
        self.x = random.randint(0, SCREEN_WIDTH // BLOCK_SIZE - 1) * BLOCK_SIZE
        self.y = random.randint(0, SCREEN_HEIGHT // BLOCK_SIZE - 1) * BLOCK_SIZE
        self.rectangle = pygame.Rect(self.x, self.y, BLOCK_SIZE, BLOCK_SIZE)

    def update(self):
        self.rectangle = pygame.Rect(self.x, self.y, BLOCK_SIZE, BLOCK_SIZE)
