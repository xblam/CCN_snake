# Snake: Deep Convolutional Q-Learning - Environment file

import numpy as np
import pygame as pg
from collections import deque


get_dir = {
    "UP" : 0,
    "DOWN" : 1,
    "RIGHT" : 2,
    "LEFT" : 3
}

display = False
class Environment():
    
    def __init__(self):
        self.snake = deque()
        self.width = 880
        self.height = 880
        self.nRows = 8
        self.nColumns = 8
        self.initSnakeLen = 2
        self.stepReward = -0.03
        self.deathReward = -1.
        self.foodReward = 2.

        self.screen = pg.display.set_mode((self.width, self.height))

        self.reset()
        
        # screenmap is the same layout i had as the matrix
        
    

    def reset(self):

        self.snake.clear()
        self.screenMap = np.zeros((self.nRows, self.nColumns))
        
        # draw in the snake's initial position
        self.snake.append((int(self.nRows / 2), int(self.nColumns / 2)))
        self.snake.append((int(self.nRows/2) + 1, int(self.nColumns/2)))

        # spawn in the fruit
        self.fruitPos = self.placeFruit()

        self.update_screenMap()

        self.collected = False
        
        self.lastMove = "UP"
        
        self.drawScreen()
    # def update_screenMap(self):
    #     for square in self.snake:
    #         self.screenMap[int(self.nRows / 2) + 1][int(self.nColumns / 2)] = 0.5
    
    def placeFruit(self):
        posx = np.random.randint(0, self.nColumns)
        posy = np.random.randint(0, self.nRows)
        while self.screenMap[posy][posx] == 0.5:
            posx = np.random.randint(0, self.nColumns)
            posy = np.random.randint(0, self.nRows)
        
        self.screenMap[posy][posx] = 1
        
        return (posy, posx)
    
    def update_screenMap(self):
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
                    pg.draw.rect(self.screen, (255, 255, 255), (j*cellWidth + 1, i*cellHeight + 1, cellWidth - 2, cellHeight - 2))
                elif self.screenMap[i][j] == 1:
                    pg.draw.rect(self.screen, (255, 0, 0), (j*cellWidth + 1, i*cellHeight + 1, cellWidth - 2, cellHeight - 2))
                    
        pg.display.flip()
      
    def moveSnake(self, nextPos, col):
        
        self.snake.appendleft(nextPos)
        
        if not col:
            self.snake.pop()
        
        self.screenMap = np.zeros((self.nRows, self.nColumns))
        
        self.update_screenMap()
        
        if col:
            self.fruitPos = self.placeFruit()
            self.collected = True
            
        self.screenMap[self.fruitPos[0]][self.fruitPos[1]] = 1
        
    def step(self, direction):
        # direction = 0 -> up
        # direction = 1 -> down
        # direction = 2 -> right
        # direction = 3 -> left
        gameOver = False
        reward = self.stepReward
        self.collected = False
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
        
        snakeX = self.snake[0][1]
        snakeY = self.snake[0][0]
        
        if direction == "UP" and self.lastMove == "DOWN":
            direction = "DOWN"
        if direction == "DOWN" and self.lastMove == "UP":
            direction = "UP"
        if direction == "LEFT" and self.lastMove == "RIGHT":
            direction = "RIGHT"
        if direction == "RIGHT" and self.lastMove == "LEFT":
            direction = "LEFT"
        
        if direction == "UP":
            if snakeY > 0:
                if self.screenMap[snakeY - 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.deathReward
                elif self.screenMap[snakeY - 1][snakeX] == 1:
                    reward = self.foodReward
                    self.moveSnake((snakeY - 1, snakeX), True)
                elif self.screenMap[snakeY - 1][snakeX] == 0:
                    self.moveSnake((snakeY - 1, snakeX), False)
            else:
                gameOver = True
                reward = self.deathReward
                
        elif direction == 1:
            if snakeY < self.nRows - 1:
                if self.screenMap[snakeY + 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.deathReward
                elif self.screenMap[snakeY + 1][snakeX] == 1:
                    reward = self.foodReward
                    self.moveSnake((snakeY + 1, snakeX), True)
                elif self.screenMap[snakeY + 1][snakeX] == 0:
                    self.moveSnake((snakeY + 1, snakeX), False)
            else:
                gameOver = True
                reward = self.deathReward
                
        elif direction == 2:
            if snakeX < self.nColumns - 1:
                if self.screenMap[snakeY][snakeX + 1] == 0.5:
                    gameOver = True
                    reward = self.deathReward
                elif self.screenMap[snakeY][snakeX + 1] == 1:
                    reward = self.foodReward
                    self.moveSnake((snakeY, snakeX + 1), True)
                elif self.screenMap[snakeY][snakeX + 1] == 0:
                    self.moveSnake((snakeY, snakeX + 1), False)
            else:
                gameOver = True
                reward = self.deathReward 
        
        elif direction == 3:
            if snakeX > 0:
                if self.screenMap[snakeY][snakeX - 1] == 0.5:
                    gameOver = True
                    reward = self.deathReward
                elif self.screenMap[snakeY][snakeX - 1] == 1:
                    reward = self.foodReward
                    self.moveSnake((snakeY, snakeX - 1), True)
                elif self.screenMap[snakeY][snakeX - 1] == 0:
                    self.moveSnake((snakeY, snakeX - 1), False)
            else:
                gameOver = True
                reward = self.deathReward
                
        self.drawScreen()
        
        self.lastMove = direction
        
        pg.time.wait(1)
        
        return self.screenMap, reward, gameOver
            
            


if __name__ == '__main__':
    env = Environment()
    gameOver = False
    start = False
    direction = "UP"
    while True:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE and not start:
                    start = True
                elif event.key == pg.K_SPACE and start:
                    start = False
                if event.key == pg.K_UP:
                    direction = "UP"
                elif event.key == pg.K_DOWN:
                    direction = "DOWN"
                elif event.key == pg.K_RIGHT:
                    direction = "RIGHT"
                elif event.key == pg.K_LEFT:
                    direction = "LEFT"
        
        if start:
            _, _, gameOver = env.step(direction)
            
        if gameOver:
            start = False
            gameOver = False
            env.reset()
            direction = "UP"
                
              
