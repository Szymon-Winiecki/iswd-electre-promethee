import numpy as np

from graph_drawing import draw_complete_preorder, draw_preorder, print_complete_preorder
from ElectreSolver import ElectreSolver

class Electre_III_Solver(ElectreSolver):
    def __init__(self, criteria, data):
        super().__init__(criteria, data)

    def solve(self):
        self.calc_marginal_concordance_discordance()
        self.aggregate_marginal_concordance_discordance()

        self.desc_preorder = self.distilate(
            self.outranking_credibility, 
            np.arange(self.outranking_credibility.shape[0]),
            'desc'
        )

        self.asc_preorder = self.distilate(
            self.outranking_credibility, 
            np.arange(self.outranking_credibility.shape[0]),
            'asc'
        )

        desc_ranks = np.zeros(self.outranking_credibility.shape[0])
        for r in range(len(self.desc_preorder)):
            for variant in self.desc_preorder[r]:
                desc_ranks[variant] = r

        asc_ranks = np.zeros(self.outranking_credibility.shape[0])
        for r in range(len(self.asc_preorder)):
            for variant in self.asc_preorder[r]:
                asc_ranks[variant] = r

        desc_ranking_relation = np.sign(np.tile(desc_ranks, (desc_ranks.shape[0], 1)) - np.transpose(np.tile(desc_ranks, (desc_ranks.shape[0], 1))))
        asc_ranking_relation = np.sign(np.tile(asc_ranks, (asc_ranks.shape[0], 1)) - np.transpose(np.tile(asc_ranks, (asc_ranks.shape[0], 1))))

        final_ranking_relation = np.sign((desc_ranking_relation + asc_ranking_relation) + (desc_ranking_relation * asc_ranking_relation))

        final_preorder = self.relation_to_preorder(final_ranking_relation)

        self.final_ranks = self.preorder_to_ranks(final_preorder)

        self.calc_median_preorder()

        print_complete_preorder(self.final_ranks, "results/electreIII", "final_ranks.txt")
        print_complete_preorder(self.median_preorder, "results/electreIII", "median_ranks.txt")


        draw_complete_preorder(self.desc_preorder, "results/electreIII", "desc_preorder.png")
        draw_complete_preorder(self.asc_preorder, "results/electreIII", "asc_preorder.png")
        draw_preorder(final_preorder, "results/electreIII", "final_preorder.png")

    def calc_median_preorder(self):

        def rank_in(variant, preorder): 
            rank = 0
            while variant not in preorder[rank]:
                rank += 1
            return rank
        
        def resolve(variants):
            rank_sum = np.zeros((len(variants)))
            for i in range(len(variants)):
                rank_sum[i] = rank_in(variants[i], self.asc_preorder) + rank_in(variants[i], self.desc_preorder)
            sorted_idx = np.argsort(rank_sum)
            rank_sum = rank_sum[sorted_idx]
            sorted_variants =  np.array(variants)[sorted_idx]
            resolved_preorder = [[sorted_variants[0]]]
            for i in range(1, len(sorted_variants)):
                if rank_sum[i] != rank_sum[i-1]:
                    resolved_preorder.append([])
                resolved_preorder[-1].append(sorted_variants[i])

            return resolved_preorder
            

        self.median_preorder = []

        for rank in self.final_ranks:
            if len(rank) == 1:
                self.median_preorder.append(rank)
            else:
                for r in resolve(rank):
                    self.median_preorder.append(r)
