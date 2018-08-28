from collections import namedtuple
import itertools as it
import os
from random import sample as rsample
import time

import numpy as np

from sklearn.neural_network import MLPClassifier

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import SGD, RMSprop, Adam

from matplotlib import pyplot as plt

## Creating the Game
class Snake(object):
    def __init__(self, rewards, grid_size):
        self.grid_size = grid_size
        self.snake_length = 3
        self.Fruit = namedtuple('Fruit', ['x', 'y'])
        self.life_reward = rewards[0]
        self.alive_reward = rewards[1]
        self.death_reward = rewards[2]
        self.reset()
        
    def reset(self):
        self.actions = [(-1, 0)] * self.snake_length  # An action for each snake segment
        self.head_x = self.grid_size // 2 - self.snake_length // 2
        self.snake = [(x, self.grid_size // 2) for x in range(self.head_x, self.head_x + self.snake_length)]
        self.grow = -1  # Don't start growing snake yet
        self.fruit = self.Fruit(-1, -1)
        
    def play(self):
        self.reset()
        while True:
            # Draw borders
            screen = np.zeros((self.grid_size, self.grid_size))
            screen[[0, -1]] = 1
            screen[:, [0, -1]] = 1
            sum_of_borders = screen.sum()

            # Draw snake
            for segm in self.snake:
                x, y = segm
                screen[y, x] = 1

            # Snake hit into wall or ate itself
            end_of_game = len(self.snake) > len(set(self.snake)) or screen.sum() < sum_of_borders + len(self.snake)
            reward = self.death_reward * end_of_game if end_of_game else self.alive_reward

            # Draw fruit
            if screen[self.fruit.y, self.fruit.x] > .5:
                self.grow += 1
                reward = len(self.snake) * self.life_reward
                while True:
                    self.fruit = self.Fruit(*np.random.randint(1, self.grid_size - 1, 2))
                    if screen[self.fruit.y, self.fruit.x] < 1:
                        break

            screen[self.fruit.y, self.fruit.x] = .5

            action = yield screen, reward, len(self.snake)-self.snake_length

            step_size = sum([abs(act) for act in action])
            if not step_size:
                action = self.actions[0]  # Repeat last action
            elif step_size > 1:
                raise ValueError('Cannot move more than 1 unit at a time')

            self.actions.insert(0, action)
            self.actions.pop()

            # For as long as the snake needs to grow,
            # copy last segment, and add (0, 0) action
            if self.grow > 0:
                self.snake.append(self.snake[-1])
                self.actions.append((0, 0))
                self.grow -= 1

            # Update snake segments
            for ix, act in enumerate(self.actions):
                x, y = self.snake[ix]
                delta_x, delta_y = act
                self.snake[ix] = x + delta_x, y + delta_y

            if end_of_game:
                break

## Agent Model
class Agent(object):
    def __init__(self, 
                 all_possible_actions,
                 gamma=0.9, 
                 batch_size=32,
                 epsilon=1,
                 nb_frames = 4,
                 grid_size=10,
                 rewards=[5, -1, -10],
                 load_path='',
                 save_path=''):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.min_epsilon = 0.3
        self.epsilon_rate = 0.99
        self.action_set = all_possible_actions
        self.nb_actions = len(self.action_set)
        self.rewards = rewards
        self.nb_frames = nb_frames
        self.save_path = save_path
        
        self.grid_size = grid_size

        self.model = self.build_model(load_path)
        
        self.env = Snake(self.rewards, self.grid_size)
        
    def build_model(self, load_path):
        num_filters = [16, 32]
        
        model = Sequential()
        model.add(BatchNormalization(axis=1, input_shape=(self.nb_frames, self.grid_size, self.grid_size)))
        for filters in num_filters:
            model.add(Conv2D(filters=filters, 
                             input_shape = (self.nb_frames, self.grid_size, self.grid_size), 
                             kernel_size=(3,3), 
                             padding='same', 
                             activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.nb_actions, activation='softmax'))
                       
        if load_path!='':
            model.load_weights(load_path)
        model.compile(optimizer=SGD(lr=0.001), loss='mse', metrics=['accuracy'])
        
        return model
    
    def save_weights(self):
        if self.save_path!='':
            self.model.save_weights(self.save_path, overwrite=True)
    
    def model_summary(self):
        print(self.model.summary())
    
    def experience_replay(self, batch_size):
        """
        Coroutine of experience replay.

        Provide a new experience by calling send, which in turn yields 
        a random batch of previous replay experiences.
        """
        memory = []
        while True:
            experience = yield rsample(memory, batch_size) if batch_size <= len(memory) else None
            memory.append(experience)
    

    def train(self, nb_epochs=1000):
        self.exp_replay = self.experience_replay(self.batch_size)
        # Start experience replay coroutine
        next(self.exp_replay)
        
        for i in range(nb_epochs):
            g = self.env.play()
            screen, _, _ = next(g)
            S = np.asarray([screen] * self.nb_frames)
            try:
                # Decrease epsilon over the first half of training
                if self.epsilon > self.min_epsilon:
                    self.epsilon -= self.epsilon_rate / (nb_epochs / 2)

                while True:
                    if np.random.random() < self.epsilon:
                        ix = np.random.randint(self.nb_actions)
                    else:
                        ix = np.argmax(self.model.predict(S[np.newaxis]), axis=-1)[0]

                    action = self.action_set[ix]
                    screen, reward, _ = g.send(action)
                    S_prime = np.zeros_like(S) 
                    S_prime[1:] = S[:-1]
                    S_prime[0] = screen
                    experience = (S, action, reward, S_prime)
                    S = S_prime

                    batch = self.exp_replay.send(experience)

                    if batch:
                        inputs = []
                        targets = []
                        for s, a, r, s_prime in batch:
                            # The targets of unchosen actions are set to the q-values of the model,
                            # so that the corresponding errors are 0. The targets of chosen actions
                            # are set to either the rewards, in case a terminal state has been reached, 
                            # or future discounted q-values, in case episodes are still running.
                            t = self.model.predict(s[np.newaxis]).flatten()
                            ix = self.action_set.index(a)
                            if r < 0:
                                t[ix] = r
                            else:
                                t[ix] = r + self.gamma * self.model.predict(s_prime[np.newaxis]).max(axis=-1)
                            targets.append(t)
                            inputs.append(s)

                        self.model.train_on_batch(np.array(inputs), np.array(targets))    

            except StopIteration:
               pass
            
            if i==0 or (i+1) % 1000 == 0:
                print('Epoch %6i/%i, epsilon: %.3f' % (i + 1, nb_epochs, self.epsilon))
                self.save_weights()
        
        print('Training complete..\n')

    def render(self, render=False, save=False):
        if save:
            if 'images' not in os.listdir('.'):
                os.mkdir('images')
        
        frame_cnt = it.count()
        while True:
            screen = (yield)
            if render:
                clear_output(wait=True)
                plt.imshow(screen, interpolation='none', cmap='gray')
                display(plt.show())
            
            if save:
                plt.imshow(screen, interpolation='none', cmap='gray')
                plt.savefig('images/%04i.png' % (next(frame_cnt), ))
    
    
    def test(self, render, nb_episodes=10):
        img_saver = self.render(render)
        next(img_saver)
        
        scores = []
        self.max_episode_length = 100
        
        for _ in range(nb_episodes):
            alive_reward_cap = 0

            g = self.env.play()
            screen, _, init_score = next(g)
            img_saver.send(screen)
            frame_cnt = it.count()
            try:
                S = np.asarray([screen] * self.nb_frames)
                while True:
                    next(frame_cnt)
                    ix = np.argmax(self.model.predict(S[np.newaxis]), axis=-1)[0]
                    screen, r, score = g.send(self.action_set[ix])
                    S[1:] = S[:-1]
                    S[0] = screen
                    img_saver.send(screen)

                    if r % 5 == 0:
                        alive_reward_cap = 0
                    elif r == -1:
                        alive_reward_cap += 1

                    if alive_reward_cap > self.max_episode_length * (score+1):
                        raise StopIteration

            except StopIteration:
                scores.append(score)
                
        img_saver.close()
        return scores

## Playing
def playSnake(training_iterations=1000, 
              test_episodes=10,
              show_model=False):
    
    params = dict(
        all_possible_actions=((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
                     gamma=0.8, 
                     batch_size=64,
                     epsilon=1,
                     nb_frames = 2,
                     grid_size=10,
                     rewards=[5, -1, -10],
                    load_path='game_weights/snake_game_weights_v2.h5',
                    save_path=''
    )

    target_score = 8
    max_scores = []
    attempts = 2
    
    agent = Agent(**params)
    
    if show_model:
        agent.model_summary()
    
    for attempt in range(attempts):
        # ====Testing the model====
        scores = agent.test(render=False, nb_episodes=test_episodes)
        max_scores.append(max(scores))
        
        if max(scores) == target_score:
            print("\n==========\nTarget achieved successfully!\n==========")
            plt.bar(range(len(scores)),scores)
            break
        
        if max(scores) <= 5:
            # ====Training the model====
            print('\n-----Commencing Training process-----')
            agent.train(nb_epochs=training_iterations)
            
    if attempts > 1:
        plt.title('Max. Scores per iteration')
        plt.plot(max_scores, label='Max scores per attempt')
    else:
        plt.title('Score summary per episode')
        plt.bar(range(len(scores)),scores)


playSnake(training_iterations=5000, test_episodes=50)

playSnake(training_iterations=2000, test_episodes=50)

playSnake(test_episodes=100)

