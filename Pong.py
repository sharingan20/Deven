import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def build_model(learning_rate, input_shape, action_size):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(action_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))

    model.summary()
    return model


class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=2000)

        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 100

        self.model = build_model(self.learning_rate, self.state_size, self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            #state = np.reshape(state, [1, ])
            state = np.array([state])
            #print(state)
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        sample_batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, done = zip(*sample_batch)
        states = np.array(states)
        next_states = np.array(next_states)
        
        targets = self.model.predict(states)
        next_targets = self.model.predict(next_states)
        
        for i in range(self.batch_size):
            if done[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_targets[i])

        self.model.fit(states, targets, batch_size=self.batch_size, verbose=0)


class Pong:

    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        
        self.episodes = 1000
        self.agent = Agent(self.state_size, self.action_size)

    def run(self):
        for e in range(self.episodes):
            state = self.env.reset() / 255.0
            done = False
            #print('here1')
            for i in range(700):
                self.env.render()
                print(i)
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                #print('here2')
                next_state = next_state / 255.0
                #print(action)
                self.agent.remember(state, action, reward, next_state, done)
                #print('here3')                
                state = next_state
                if done:
                    print(f"episode: {e}/{self.episodes},score: {i}, e: {self.agent.epsilon:.2}")
                    if i == 499:
                        print("Save")
                        return
                    break
                self.agent.replay()
            
def main():
    pong = Pong()
    pong.run()

if __name__ == "__main__":
    main()
