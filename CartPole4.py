import os
import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam



class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.replay_memory = deque(maxlen=10000)
        self.min_replay_memory = 500 #1000

        self.learning_rate = 0.001 #0.00025
        self.gamma = 0.99
        self.epsilon = 0.2 #1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64

        self.model = self.create_model(self.learning_rate, self.state_size, self.action_size)

        self.target_model = self.create_model(self.learning_rate, self.state_size, self.action_size)
        self.target_model.set_weights(self.model.get_weights())

        self.target_update_counter = 0
        self.target_update_every = 5


    def create_model(self, learning_rate, input_shape, action_size):
        model = Sequential()

        model.add(Dense(256, input_shape=input_shape, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

        model.summary()
        return model


    def update_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.reshape(state, [-1, *self.state_size])
            return np.argmax(self.model.predict(state))

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replay_memory:
            return

        sample_batch = random.sample(self.replay_memory, self.batch_size)

        states, actions, rewards, next_states, done = zip(*sample_batch)
        states = np.array(states)
        next_states = np.array(next_states)
        
        current_qs_list = self.model.predict(states)
        next_qs_list = self.target_model.predict(next_states)
        
        for i in range(self.batch_size):
            if done[i]:
                current_qs_list[i][actions[i]] = rewards[i]
            else:
                current_qs_list[i][actions[i]] = rewards[i] + self.gamma * np.max(next_qs_list[i])

        self.model.fit(states, current_qs_list, batch_size=self.batch_size, verbose=0)

        if terminal_state:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.target_update_counter += 1

        if self.target_update_counter > self.target_update_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class CartPole:

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        self.episodes = 2000
        self.agent = Agent(self.state_size, self.action_size)

    def load(self, name):
        self.agent.model = load_model(name)

    def save(self, name):
        self.agent.model.save(name)

    def run(self):
        for e in range(self.episodes):
            state = self.env.reset()
            done = False
            i = 0

            while not done:
                self.env.render()
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.update_replay_memory(state, action, reward, next_state, done)
                state = next_state
                self.agent.train(done)
                i += 1
            else:
                print(f"episode: {e}/{self.episodes}, score: {i}, epsilon: {self.agent.epsilon:.4}")
                if i == 500:
                    print("Successfully completed")
                    #self.save("cartpole-dqn.h5")
                    #return


    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(3):#self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [-1, *self.state_size])
            done = False
            i = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [-1, *self.state_size])
                i += 1
            else:
                print(f"episode: {e}/{self.episodes},score: {i}")
        
def main():
    cartpole = CartPole()
    cartpole.run()
    #cartpole.test()

if __name__ == "__main__":
    main()
