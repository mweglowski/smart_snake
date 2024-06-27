import sys
import pygame
import numpy as np
import matplotlib.pyplot as plt
from snake import Snake
from agent import Agent
from utils import get_danger_state, get_food_state

BLOCK_SIZE = 50
SCREEN_WIDTH = BLOCK_SIZE * 10
SCREEN_HEIGHT = BLOCK_SIZE * 10

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.state_size = 12  # Updated state size
        self.action_size = 4  # 4 possible actions: up, down, left, right
        self.agent = Agent(self.state_size, self.action_size)
        self.batch_size = 32
        self.mode = None
        self.scores = []

        # Initialize matplotlib for interactive plotting
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Score')
        self.ax.set_title('Training Progress')

    def draw_rectangle(self, rectangle, color="cyan"):
        pygame.draw.rect(self.screen, color, rectangle)

    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, BLOCK_SIZE):
            for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
                rectangle = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.screen, "#222222", rectangle, 1)

    def get_state(self):
        direction = [self.snake.x_direction == 1, self.snake.x_direction == -1, self.snake.y_direction == 1, self.snake.y_direction == -1]
        danger = get_danger_state(self.snake)
        food = get_food_state(self.snake)
        state = direction + danger + food
        state = np.reshape(state, [1, self.state_size])
        
        return state

    def run(self):
        self.wait_for_mode_selection()
        
        if self.mode == 'a':
            self.train_and_visualize_agent(epochs=1000)
        elif self.mode == 'u':
            self.user_play()
        else:
            pygame.quit()
            sys.exit()

    def wait_for_mode_selection(self):
        while self.mode is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.mode = 'q'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_u:
                        self.mode = 'u'
                    elif event.key == pygame.K_a:
                        self.mode = 'a'
                    elif event.key == pygame.K_q:
                        self.mode = 'q'

            self.screen.fill("black")
            
            font = pygame.font.Font(None, 36)
            text = font.render("Press 'u' to play, 'a' for agent, 'q' to quit", True, (255, 255, 255))
            
            self.screen.blit(text, (20, SCREEN_HEIGHT // 2))
            pygame.display.flip()
            self.clock.tick(10)

    def user_play(self):
        running = True
        
        self.snake = Snake()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        if self.snake.y_direction != -1:
                            self.snake.y_direction = 1
                            self.snake.x_direction = 0
                    elif event.key == pygame.K_w:
                        if self.snake.y_direction != 1:
                            self.snake.y_direction = -1
                            self.snake.x_direction = 0
                    elif event.key == pygame.K_a:
                        if self.snake.x_direction != 1:
                            self.snake.y_direction = 0
                            self.snake.x_direction = -1
                    elif event.key == pygame.K_d:
                        if self.snake.x_direction != -1:
                            self.snake.y_direction = 0
                            self.snake.x_direction = 1

            terminal, reward = self.snake.step()
            if terminal:
                self.snake = Snake()
                print(f"Score: {self.snake.score}")
                # running = False

            self.update_screen()
            self.clock.tick(8)

        pygame.quit()
        sys.exit()

    def agent_play_step(self):
        state = self.get_state()
        action = self.agent.act(state)

        # Convert action to direction
        if action == 0 and self.snake.y_direction != 1:
            self.snake.y_direction = -1
            self.snake.x_direction = 0
        elif action == 1 and self.snake.y_direction != -1:
            self.snake.y_direction = 1
            self.snake.x_direction = 0
        elif action == 2 and self.snake.x_direction != 1:
            self.snake.y_direction = 0
            self.snake.x_direction = -1
        elif action == 3 and self.snake.x_direction != -1:
            self.snake.y_direction = 0
            self.snake.x_direction = 1

        terminal, reward = self.snake.step()
        # reward = -10 if terminal else 1
        next_state = self.get_state()

        self.agent.remember(state, action, reward, next_state, terminal)

        if len(self.agent.memory) > self.batch_size:
            self.agent.replay(self.batch_size)

        if terminal:
            print(f"Score: {self.snake.score}")
            self.scores.append(self.snake.score)
            self.update_plot(len(self.scores))
            self.snake = Snake()  # Reset snake

        self.update_screen()

    def train_and_visualize_agent(self, epochs):
        # Load existing model if exists
        try:
            # self.agent.load_model("snake_model.pth")
            print("Model loaded!")
        except:
            print("No existing model found, training...")
        
        for e in range(1000):
            self.snake = Snake()  # Reset snake for each epoch
            state = self.get_state()
            done = False
            while not done:
                action = self.agent.act(state)
                
                # Perform action
                if action == 0 and self.snake.y_direction != 1:
                    self.snake.y_direction = -1
                    self.snake.x_direction = 0
                elif action == 1 and self.snake.y_direction != -1:
                    self.snake.y_direction = 1
                    self.snake.x_direction = 0
                elif action == 2 and self.snake.x_direction != 1:
                    self.snake.y_direction = 0
                    self.snake.x_direction = -1
                elif action == 3 and self.snake.x_direction != -1:
                    self.snake.y_direction = 0
                    self.snake.x_direction = 1
                
                done, reward = self.snake.step()
                # Change to include reward by eating food, because currently snake tries to survive as long as possible
                reward = -10 if done else 1
                next_state = self.get_state()

                self.agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)

                self.update_screen()
                self.clock.tick(100)  # Control the game speed

            print(f"Epoch {e+1}/{epochs} - Score: {self.snake.score}")
            self.scores.append(self.snake.score)
            self.update_plot(e+1)
            print(self.snake.episode_reward)
            self.snake.episode_reward = 0
            
        # Saving model
        self.agent.save_model("snake_model.pth")
        print("Model saved")

    def update_screen(self):
        # Fill background with black and draw gridlines
        self.screen.fill("black")
        self.draw_grid()

        # Update new food position
        self.snake.food.update()
        
        # Draw food
        self.draw_rectangle(self.snake.food.rectangle, "yellow")

        # Draw snake
        self.draw_rectangle(self.snake.head)
        for body_part in self.snake.body:
            self.draw_rectangle(body_part)

        pygame.display.update()

    def update_plot(self, epoch):
        self.line.set_xdata(np.arange(len(self.scores)))
        self.line.set_ydata(self.scores)
        self.ax.set_xlim(0, max(10, len(self.scores)))
        self.ax.set_ylim(0, max(10, max(self.scores) + 10))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

game = Game()
game.run()
