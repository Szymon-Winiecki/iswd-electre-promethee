import numpy as np

from Solver import Solver
from Criterion import Criterion

class PrometheeSolver(Solver):
    def __init__(self, criteria, data):
        self.criteria = [Criterion(*c) for c in criteria]
        self.data = data
        self.marginal_preference_matrix = np.zeros((data.shape[0], data.shape[0], len(criteria)))
        self.preference_matrix = np.zeros((data.shape[0], data.shape[0]))
        self.positive_flow = np.zeros((data.shape[0]))
        self.negative_flow = np.zeros((data.shape[0]))
    
    def calc_marginal_preferances(self):
        b = np.tile(self.data[np.newaxis, :], (self.data.shape[0], 1, 1))
        a = np.transpose(b, axes=(1,0,2))

        diff = a-b
        dir = np.array([c.type for c in self.criteria])

        diff = diff * dir
        
        q = np.array([c.q for c in self.criteria])
        p = np.array([c.p for c in self.criteria])

        lq = diff <= q
        gp = diff >= p

        self.marginal_preference_matrix = (diff - q) / (p - q)
        self.marginal_preference_matrix[lq] = 0
        self.marginal_preference_matrix[gp] = 1

    def aggregate_marginal_preferences(self):
        weights = np.array([c.weight for c in self.criteria])
        self.preference_matrix = (self.marginal_preference_matrix * weights).sum(axis=2) / weights.sum()

    def calc_flow(self):
        self.positive_flow = self.preference_matrix.sum(axis=1)
        self.negative_flow = self.preference_matrix.sum(axis=0)

    def calc_ranking(self):
        pass

    def flow_to_complete_preorder(self, flow):
        sort_idx = np.argsort(flow)

        preorder = [[sort_idx[0]]]

        i = 1
        last_val = None
        while i < sort_idx.shape[0]:
            if flow[sort_idx[i]] != flow[sort_idx[i-1]]:
                preorder.append([])
            preorder[-1].append(sort_idx[i])
            i += 1

        return preorder


    def preference(self, a, b, criterion):
        diff = 0

        if self.criteria[criterion].type == 1: # gain
            diff = self.data[a][criterion] - self.data[b][criterion]
        else: # cost
            diff = self.data[b][criterion] - self.data[a][criterion]

        q = self.criteria[criterion].q
        p = self.criteria[criterion].p

        if diff <= q:
            return 0
        
        if diff >= p:
            return 1
        
        return (diff - q) / (p - q)