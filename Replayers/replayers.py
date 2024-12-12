### Updated replayers.py ###

import numpy as np
from tqdm import tqdm

class ReplaySimulator(object):
    def __init__(self, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1, random_seed=1):
        np.random.seed(random_seed)
        self.reward_history = reward_history
        self.item_col_name = item_col_name
        self.visitor_col_name = visitor_col_name
        self.reward_col_name = reward_col_name
        self.n_visits = n_visits
        self.n_iterations = n_iterations
        self.items = self.reward_history[self.item_col_name].unique()
        self.n_items = len(self.items)
        self.visitors = self.reward_history[self.visitor_col_name].unique()
        self.n_visitors = len(self.visitors)

    def reset(self):
        self.n_item_samples = np.zeros(self.n_items)
        self.n_item_rewards = np.zeros(self.n_items)

    def replay(self):
        results = []
        for iteration in tqdm(range(0, self.n_iterations)):
            self.reset()
            total_rewards = 0
            for visit in range(0, self.n_visits):
                found_match = False
                while not found_match:
                    visitor_idx = np.random.randint(self.n_visitors)
                    visitor_id = self.visitors[visitor_idx]
                    item_idx = self.select_item()
                    item_id = self.items[item_idx]
                    reward = self.reward_history.query(
                        '{} == @item_id and {} == @visitor_id'.format(self.item_col_name, self.visitor_col_name))[self.reward_col_name]
                    found_match = reward.shape[0] > 0
                reward_value = reward.iloc[0]
                self.record_result(visit, item_idx, reward_value)
                total_rewards += reward_value
                results.append({'visit': visit, 'fraction_relevant': total_rewards / (visit + 1)})
        return results

    def select_item(self):
        return np.random.randint(self.n_items)

    def record_result(self, visit, item_idx, reward):
        self.n_item_samples[item_idx] += 1
        alpha = 1./self.n_item_samples[item_idx]
        self.n_item_rewards[item_idx] += alpha * (reward - self.n_item_rewards[item_idx])

class ABTestReplayer(ReplaySimulator):
    def __init__(self, n_visits, n_test_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(ABTestReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        self.n_test_visits = n_test_visits
        self.is_testing = True
        self.best_item_id = None

    def reset(self):
        super(ABTestReplayer, self).reset()
        self.is_testing = True
        self.best_item_idx = None

    def select_item(self):
        if self.is_testing:
            return super(ABTestReplayer, self).select_item()
        else:
            return self.best_item_idx

    def record_result(self, visit, item_idx, reward):
        super(ABTestReplayer, self).record_result(visit, item_idx, reward)
        if visit == self.n_test_visits - 1:
            self.is_testing = False
            self.best_item_idx = np.argmax(self.n_item_rewards)

class EpsilonGreedyReplayer(ReplaySimulator):
    def __init__(self, epsilon, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(EpsilonGreedyReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        self.epsilon = epsilon

    def select_item(self):
        if np.random.uniform() < self.epsilon:
            return super(EpsilonGreedyReplayer, self).select_item()
        else:
            return np.argmax(self.n_item_rewards)

class ThompsonSamplingReplayer(ReplaySimulator):
    def reset(self):
        self.alphas = np.ones(self.n_items)
        self.betas = np.ones(self.n_items)

    def select_item(self):
        samples = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.argmax(samples)

    def record_result(self, visit, item_idx, reward):
        if reward == 1:
            self.alphas[item_idx] += 1
        else:
            self.betas[item_idx] += 1

class SoftmaxReplayer(ReplaySimulator):
    def __init__(self, tau, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(SoftmaxReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        self.tau = tau

    def select_item(self):
        exp_values = np.exp(self.n_item_rewards / self.tau)
        probabilities = exp_values / exp_values.sum()
        return np.random.choice(range(self.n_items), p=probabilities)


class UCBReplayer(ReplaySimulator):
    def __init__(self, c, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(UCBReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        self.c = c

    def select_item(self):
        total_counts = np.sum(self.n_item_samples)
        ucb_values = self.n_item_rewards + self.c * np.sqrt(np.log(total_counts + 1) / (self.n_item_samples + 1e-5))
        return np.argmax(ucb_values)
