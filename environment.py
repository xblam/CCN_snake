# Snake: Deep Convolutional Q-Learning - Environment file

import numpy as np
import pygame as pygame
from collections import deque

get_reward = {
    0 : -0.03,
    1 : -1,
    2 : 2,
}

display = False
class Environment():
    
    def __init__(self):
        self.snake = deque()
        self.width = 880
        self.height = 880
        self.nRows = 8
        self.nColumns = 8
        self.stepReward = -0.03
        self.deathReward = -1.
        self.foodReward = 2.

        self.screen = pygame.display.set_mode((self.width, self.height))

        self.reset()
        
        # screenmap is the same layout i had as the matrix
        
    

    def reset(self):
        self.initial_dir = 0
        self.snake.clear()
        self.screenMap = np.zeros((self.nRows, self.nColumns))
        
        # draw in the snake's initial position
        self.snake.append((int(self.nRows / 2), int(self.nColumns / 2)))
        self.snake.append((int(self.nRows/2) + 1, int(self.nColumns/2)))
        # spawn in the fruit
        self.fruitPos = self.placeFruit()

        self.update_screenMap()

        self.collected = False
        
        self.lastMove = 0
        
        self.drawScreen()

    
    def placeFruit(self):
        posx = np.random.randint(0, self.nColumns)
        posy = np.random.randint(0, self.nRows)
        # make sure that the fruit does not spawn in the body of the snake
        while self.screenMap[posy][posx] == 0.5:
            posx = np.random.randint(0, self.nColumns)
            posy = np.random.randint(0, self.nRows)
        return (posy, posx)


    def update_screenMap(self):
        self.screenMap = np.zeros((self.nRows, self.nColumns))
        for part in self.snake:
            self.screenMap[part[0]][part[1]] = 0.5
        self.screenMap[self.fruitPos[0]][self.fruitPos[1]] = 1
    

    def drawScreen(self):
        self.screen.fill((0, 0, 0))
        cellWidth = self.width / self.nColumns
        cellHeight = self.height / self.nRows
        
        for i in range(self.nRows):
            for j in range(self.nColumns):
                if self.screenMap[i][j] == 0.5:
                    pygame.draw.rect(self.screen, (255, 255, 255), (j*cellWidth + 1, i*cellHeight + 1, cellWidth - 2, cellHeight - 2))
                elif self.screenMap[i][j] == 1:
                    pygame.draw.rect(self.screen, (255, 0, 0), (j*cellWidth + 1, i*cellHeight + 1, cellWidth - 2, cellHeight - 2))
        pygame.display.flip()
      
    def check_collision(self):
        row, col = self.snake[0]
        if (row, col) == self.fruitPos:
            self.placeFruit()
            return 2, False
        # check to see if any part of the snake (excluding the head) overlaps coords, and check that we are still inside bounds
        if row < 0 or row > self.nRows-1 or col < 0 or col > self.nColumns-1 or \
            any(row == part[0] and col == part[1] for i, part in enumerate(self.snake) if i > 1):
            return 1, True
    
        # else it means that the snake can just keep moving
        self.snake.pop()

        return 0, False



    # use direction to figure out what the next coordinate of our snake will be
        # action = 0 -> up
        # action = 1 -> down
        # action = 2 -> right
        # action = 3 -> left
    def get_coord(self, direction):
        row,col = self.snake[0]
        if direction == 0:
            return (row-1,col)
        elif direction == 1:
            return (row+1,col)
        elif direction == 2:
            return (row,col+1)
        elif direction == 3:
            return (row-1,col-1)

        
    def step(self, direction):
        self.collected = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
    
        # make sure that the snake cannot go back on itself
        if direction == 0 and self.lastMove == 1:
            direction = 1
        if direction == 1 and self.lastMove == 0:
            direction = 0
        if direction == 3 and self.lastMove == 2:
            direction = 2
        if direction == 2 and self.lastMove == 3:
            direction = 3
        
        # BLAM right here we have the direction already, so we can just write a function that returns the expected
        #  coord given the current position of the snake and the direction that we are headed

        next_coord = self.get_coord(direction)
        # move the head of the snake and then see if the head of the snake has hit anything
        self.snake.appendleft(next_coord)
        result, gameOver = self.check_collision()
        reward = get_reward[result]

        if not gameOver:
            self.update_screenMap()
        self.drawScreen()
        
        self.lastMove = direction
        
        pygame.time.wait(1)
        
        return self.screenMap, reward, gameOver
            
            


if __name__ == '__main__':
    env = Environment()
    gameOver = False
    start = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not start:
                    start = True
                elif event.key == pygame.K_SPACE and start:
                    start = False
                if event.key == pygame.K_UP:
                    direction = 0
                elif event.key == pygame.K_DOWN:
                    direction = 1
                elif event.key == pygame.K_RIGHT:
                    direction = 2
                elif event.key == pygame.K_LEFT:
                    direction = 3
        
        if start:
            _, _, gameOver = env.step(env.initial_dir)
            
        if gameOver:
            start = False
            gameOver = False
            env.reset()
                
              
