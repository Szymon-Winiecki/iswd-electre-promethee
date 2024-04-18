import numpy as np

from graph_drawing import draw_preorder
from PrometheeSolver import PrometheeSolver

class Promethee_I_Solver(PrometheeSolver):
    def __init__(self, criteria, data):
        super().__init__(criteria, data)

    def solve(self):
        self.calc_marginal_preferances()
        self.aggregate_marginal_preferences()
        self.calc_flow()
        self.calc_ranking()

    def calc_ranking(self):
        super().calc_ranking()

        positive_ranking_relation = np.sign(np.transpose(np.tile(self.positive_flow, (self.positive_flow.shape[0], 1))) - np.tile(self.positive_flow, (self.positive_flow.shape[0], 1)))
        negative_ranking_relation = np.sign(np.tile(self.negative_flow, (self.negative_flow.shape[0], 1)) - np.transpose(np.tile(self.negative_flow, (self.positive_flow.shape[0], 1))))

        final_ranking_relation = np.sign((positive_ranking_relation + negative_ranking_relation) + (positive_ranking_relation * negative_ranking_relation))

        self.positive_preorder = self.relation_to_preorder(positive_ranking_relation)
        self.negative_preorder = self.relation_to_preorder(negative_ranking_relation)
        self.final_preorder = self.relation_to_preorder(final_ranking_relation)

        draw_preorder(self.positive_preorder, "results/prometheeI", "positive_ranking.png")
        draw_preorder(self.negative_preorder, "results/prometheeI", "negative_ranking.png")
        draw_preorder(self.final_preorder, "results/prometheeI", "final_ranking.png")





