#Snake: Deep Convolutional Q-Learning - Training file

#Importing the libraries
from environment import Environment
from brain import Brain
from dqn import Dqn
import numpy as np
import matplotlib.pyplot as plt
import wandb



#Defining the parameters
learningRate = 0.0001
maxMemory = 60000
gamma = 0.9
batchSize = 32
nLastStates = 4

epsilon = 1.
epsilonDecayRate = 0.0002
minEpsilon = 0.05


filepathToSave = 'model2.h5'

#Initializing the Environment, the Brain and the Experience Replay Memory 
env = Environment()
brain = Brain((env.nColumns, env.nRows, nLastStates), learningRate)
model = brain.model
DQN = Dqn(maxMemory, gamma)

#Building a function that will reset current state and next state
def resetStates():
     
     currentState = np.zeros((1, env.nColumns, env.nRows, nLastStates))
     
     for i in range(nLastStates):
          currentState[0, :, :, i] = env.screenMap
     
     return currentState, currentState

#Starting the main loop
epoch = 0
max_epochs = 100
nCollected = 0
maxNCollected = 0
totNCollected = 0
total_reward = 0
epochs_per_log = 2
scores = list()

wandb.init(
    # Set the wandb project where this run will be logged
    project="convolutional_snake_ai",

    # Track hyperparameters and run metadata
    config={
        "learning_rate": learningRate,
        "architecture": "CNN",
        "epochs": max_epochs,
    }
)

# BLAM for some reason if I mess with this loop and change it to a conditinal while loop it doesnt work??
for i in range(max_epochs):
     epoch += 1
     
     #Resetting the Evironment and starting to play the game
     env.reset()
     currentState, nextState = resetStates()
     gameOver = False
     while not gameOver:
          
          #Selecting an new_direction to play
          if np.random.rand() <= epsilon:
               new_direction = np.random.randint(0, 3)
          else:
               qvalues = model.predict(currentState)[0]
               new_direction = np.argmax(qvalues)

          #Updating the Environment
          frame, reward, gameOver = env.step(new_direction)
          total_reward += reward

          
          
          frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
          nextState = np.append(nextState, frame, axis = 3)
          nextState = np.delete(nextState, 0, axis = 3)
          
          #Remembering new experience and training the AI
          DQN.remember([currentState, new_direction, reward, nextState], gameOver)
          inputs, targets = DQN.getBatch(model, batchSize)
          model.train_on_batch(inputs, targets)
          
          #Updating the score and current state
          if env.collected:
               nCollected += 1
          
          currentState = nextState

     

     #Updating the epsilon and saving the model
     epsilon -= epsilonDecayRate
     epsilon = max(epsilon, minEpsilon)
     
     if nCollected > maxNCollected and nCollected > 2:
          model.save(filepathToSave)
          maxNCollected = nCollected
          
     #Displaying the results
     totNCollected += nCollected
     nCollected = 0

     if epoch % epochs_per_log == 0 and epoch != 0:
          scores.append(totNCollected / epochs_per_log)
          wandb.log({
               "average_reward": total_reward / epochs_per_log,
               "average_food_collected": totNCollected / epochs_per_log,
               "epoch": epoch
          })
          total_reward = 0
          totNCollected = 0
          plt.plot(scores)
          plt.xlabel('Epoch / 100')
          plt.ylabel('Average Collected')
          plt.show()
          

     
     
     print('Epoch: ' + str(epoch) + ' Current Best: ' + str(maxNCollected) + ' Epsilon: {:.5f}'.format(epsilon))

wandb.finish()