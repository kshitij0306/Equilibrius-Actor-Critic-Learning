# Actor-Critic with Eligibility Traces
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


def softmax(pref: np.ndarray) -> np.ndarray:
  exp_p = np.exp(pref - np.max(pref))
  return exp_p / np.sum(exp_p)


from tqdm import tqdm
from itertools import product


def generate_combinations(n, order):
    combination = []
    for c in product(range(order + 1), repeat=n):
        combination.append(np.array(c))
    return combination


class ActorCritic:
    def __init__(self, env, runs, episodes, alpha, w_lambda, gamma, order, epsilon, max_ep, lows, highs):
        self.env = env
        self.runs = runs
        self.episodes = episodes
        self.alpha = alpha
        self.w_lambda = w_lambda
        self.gamma = gamma
        self.epsilon = epsilon
        self.order = order
        self.max_ep = max_ep
        self.action_map = {action:i for i, action in enumerate(range(self.env.action_space.n))}
        self.theta_w_length = self.env.action_space.n * (self.order+1) ** self.env.observation_space.shape[0]
        self.v_min = lows
        self.C = generate_combinations(len(self.env.observation_space.low), self.order)
        self.v_max = highs

    def get_indices(self, action):
        """
        Return start and end indices of for each action.
        """
        i = action * np.power(self.order+1, self.env.observation_space.shape[0])
        j = (action + 1) * np.power(self.order+1, self.env.observation_space.shape[0])
        return i, j

    def epsilon_greedy_policy(self, W, X, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()  # Exploration: choose a random action
        else:
            q = []
            for i in range(self.env.action_space.n):
                s, e = self.get_indices(i)
                q.append(np.dot(W[s:e], X))
            action = np.argmax(q)
        return action

    def fourier_basis(self, state):
        feature_size = (self.order + 1) ** len(state)
        X = np.zeros(feature_size)
        # c = generate_combinations(feature_size, self.order)
        # print('c',self.C)
        for i in range(feature_size):
            X[i] = np.cos(np.pi * np.dot(self.C[i], state))
        # print('XX', X)
        return X

    def select_action(self, state, theta, prob=False):
        p = []
        hot_X = self.allhot_encoder(state)
        for hx in hot_X:
            p.append(np.dot(theta, hx))
        p = softmax(p)

        if prob:
          return p
        return np.random.choice(range(self.env.action_space.n), p=p)

    def onehot_encoder(self, s, action):
        s = self.state_norm(s)
        X = self.fourier_basis(s)
        hot_vec = np.zeros(self.env.action_space.n * len(X))
        hot_vec[self.action_map[action]* len(X) : (self.action_map[action] + 1) * len(X)] = X
        return hot_vec

    def allhot_encoder(self, s):
        hot_X = []
        for a in range(self.env.action_space.n):
            hot_X.append(self.onehot_encoder(s, a))
        return np.array(hot_X)

    def get_v(self,state, w):
        Q = []
        for i in range(self.env.action_space.n):
            hot_X = self.onehot_encoder(state, i)
            Q.append(np.dot(w, hot_X))
        Q = np.array(Q)
        return np.max(Q)

    def state_norm(self, state):

      # print('self.env.observation_space.low', self.env.observation_space.low)
      state = np.array(state)
      res = []
      for i in range(len(self.v_min)):
          # print('state vals', state[i], v_min[i], (state[i] - v_min[i]), v_max[i], v_min[i], (v_max[i] - v_min[i]))
          res.append((state[i] - self.v_min[i]) / (self.v_max[i] - self.v_min[i]))

      return np.array(res)

    def grad_pi(self, state, theta, action):
        hot_X = self.onehot_encoder(state, action)
        p = self.select_action(state, theta, prob=True)
        allhot_X = self.allhot_encoder(state)
        gradient = hot_X
        for i in range(len(p)):
            gradient = gradient - (p[i] * allhot_X[i])
        return gradient

    def train(self):
        total_rewards = []
        total_steps = []
        total_w = []
        total_theta = []

        for run in range(self.runs):
            reward_arr = []
            step_arr = []
            action_arr = []
            w = np.zeros(self.theta_w_length)
            theta = np.zeros(self.theta_w_length)
            print('Run number:', run+1)
            for episode in tqdm(range(self.episodes)):
                state, _ = self.env.reset()
                action = self.select_action(state, theta)

                epi_rewards = 0
                epi_steps = 0

                z_weights = np.zeros(self.theta_w_length)
                z_theta = np.zeros(self.theta_w_length)

                i_hat = 1
                done = False
                while not done and epi_steps< self.max_ep:
                    next_state, reward, done, _, _ = self.env.step(action)

                    action_arr.append(action)
                    next_action = self.select_action(next_state, theta)
                    # print('actionnn', next_action)
                    next_val = self.get_v(next_state, w)
                    val = self.get_v(state, w)

                    td_error = reward + self.gamma * next_val - val
                    delta = self.onehot_encoder(state, action)

                    if done:
                        next_val = 0
                    z_weights = self.gamma * self.w_lambda * z_weights + delta
                    z_theta = self.gamma * self.w_lambda * z_theta + i_hat * self.grad_pi(state, theta, action)
                    w += self.alpha * td_error * z_weights
                    theta += self.alpha * td_error * i_hat * z_theta

                    state = next_state
                    action = next_action

                    i_hat = i_hat * self.gamma
                    epi_steps +=1
                    epi_rewards += reward

                reward_arr.append(epi_rewards)
                step_arr.append(epi_steps)
            print('Episode: ', episode, 'Rewards:', epi_rewards)

            total_rewards.append(reward_arr)
            total_steps.append(step_arr)
            total_w.append(w)
            total_theta.append(theta)

        return total_w, total_theta, total_steps, total_rewards


# if __name__ == "__main__":
#     env = gym.make("CartPole-v1")
#     runs = 1
#     episodes = 650
#     alpha = 0.001
#     lamda = 0.5
#     gamma = 0.9
#     order = 3
#     epsilon = 0.001
#     max_ep = 10000
#     # Create an instance of the ActorCritic class
#     ac_agent = ActorCritic(env=env, runs=runs, episodes=episodes, alpha=alpha, w_lambda=lamda, gamma=gamma, order=order, epsilon=epsilon, max_ep=max_ep)
#
#     # Train the Actor-Critic with Eligibility Traces
#     w, theta, total_steps, total_rewards = ac_agent.train()

