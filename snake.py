import pygame
from food import Food
from utils import is_food_in_snake, check_for_collision, get_danger_state, get_food_state

BLOCK_SIZE = 50
SCREEN_WIDTH = BLOCK_SIZE * 10
SCREEN_HEIGHT = BLOCK_SIZE * 10

class Snake:
    def __init__(self):
        self.x, self.y = BLOCK_SIZE, BLOCK_SIZE * 5
        self.x_direction = 1  # 1 - right, -1 - left, 0 - no movement
        self.y_direction = 0  # 1 - down, -1 - up, 0 - no movement
        self.head = pygame.Rect(self.x, self.y, BLOCK_SIZE, BLOCK_SIZE)
        self.body = [pygame.Rect(self.x - BLOCK_SIZE, self.y, BLOCK_SIZE, BLOCK_SIZE)]
        self.terminal = False
        self.score = 0
        self.episode_reward = 0
        self.food = Food()

    def step(self):
        # Handle terminal state
        self.terminal = check_for_collision(self, SCREEN_WIDTH, SCREEN_HEIGHT)

        # Move the body
        self.body.insert(0, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))

        # Update head based on current direction
        self.head.x += self.x_direction * BLOCK_SIZE
        self.head.y += self.y_direction * BLOCK_SIZE
        
        reward = 0.01

        # Check if food is within snake (head or body)
        if is_food_in_snake(self.food, self):
            reward = 10  # Large positive reward for eating food
            self.score += 1
             
            # Generate new food (and prevent from generating inside snake to make training more efficient)
            self.food = Food()
            while is_food_in_snake(self.food, self):
                self.food = Food()
        else:
            # Remove the last segment of the body if no food is eaten
            self.body.pop()

        if self.terminal:
            reward = -100
        
        self.episode_reward += reward

        # Return terminal state and reward
        return self.terminal, reward

    def get_state(self):
        head_x, head_y = self.head.x, self.head.y
        food_x, food_y = self.food.x, self.food.y

        direction_state = [self.x_direction == 1, self.x_direction == -1, self.y_direction == 1, self.y_direction == -1]

        danger_state = get_danger_state(self)
        food_state = get_food_state(self)
        state = direction_state + danger_state + food_state
        return state
