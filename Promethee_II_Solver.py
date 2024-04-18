import numpy as np

from graph_drawing import draw_complete_preorder
from PrometheeSolver import PrometheeSolver

class Promethee_II_Solver(PrometheeSolver):
    def __init__(self, criteria, data):
        super().__init__(criteria, data)
        self.net_flow = np.zeros((data.shape[0]))

    def solve(self):
        self.calc_marginal_preferances()
        self.aggregate_marginal_preferences()
        self.calc_flow()
        self.calc_ranking()

    def calc_flow(self):
        super().calc_flow()
        self.net_flow = self.positive_flow - self.negative_flow

    def calc_ranking(self):
        super().calc_ranking()  

        complete_ranking = self.flow_to_complete_preorder(self.net_flow)

        draw_complete_preorder(complete_ranking, "results/prometheeII", "complete_ranking.png")
