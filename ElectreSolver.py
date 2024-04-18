import numpy as np

from Solver import Solver
from Criterion import Criterion

class ElectreSolver(Solver):
    def __init__(self, criteria, data):
        self.criteria = [Criterion(*c) for c in criteria]
        self.data = data
        self.marginal_concordance_matrix = np.zeros((data.shape[0], data.shape[0], len(criteria)))
        self.marginal_discordance_matrix = np.zeros((data.shape[0], data.shape[0], len(criteria)))
        self.comprehensive_concordance = np.zeros((data.shape[0], data.shape[0]))
        self.outranking_credibility = np.zeros((data.shape[0], data.shape[0]))
        self.positive_flow = np.zeros((data.shape[0]))
        self.negative_flow = np.zeros((data.shape[0]))
    
    def calc_marginal_concordance_discordance(self):
        b = np.tile(self.data[np.newaxis, :], (self.data.shape[0], 1, 1))
        a = np.transpose(b, axes=(1,0,2))

        diff = b - a
        dir = np.array([c.type for c in self.criteria])

        diff = diff * dir
        
        q = np.array([c.q for c in self.criteria])
        p = np.array([c.p for c in self.criteria])

        lq = diff <= q
        gp = diff >= p

        self.marginal_concordance_matrix = (p - diff) / (p - q)
        self.marginal_concordance_matrix[lq] = 1
        self.marginal_concordance_matrix[gp] = 0

        veto = np.array([c.veto for c in self.criteria])

        self.marginal_discordance_matrix[:,:, veto != None] = np.clip((diff[:,:, veto != None] - p[veto != None]) / (veto[veto != None] - p[veto != None]), 0, 1)

    def aggregate_marginal_concordance_discordance(self):
        weights = np.array([c.weight for c in self.criteria])
        comprehensive_concordance = (self.marginal_concordance_matrix * weights).sum(axis=2) / weights.sum()

        ext_comprehensive_concordance = np.repeat(comprehensive_concordance[:,:,np.newaxis], self.marginal_discordance_matrix.shape[-1], axis=2)
        discordance_gc_select = self.marginal_discordance_matrix > ext_comprehensive_concordance

        mult_comp = np.ones_like(self.marginal_discordance_matrix)
        mult_comp[discordance_gc_select] = (1 - self.marginal_discordance_matrix[discordance_gc_select]) / (1 - ext_comprehensive_concordance[discordance_gc_select])

        self.outranking_credibility = comprehensive_concordance * np.prod(mult_comp, axis=2)


    def distilate(self, outranking_credibility, variants, dir, alpha=-0.15, beta=0.3, inherit_credibility_th=None):
        
        preorder = []
        current_variants = variants
        current_outranking_credibility = outranking_credibility * (1 - np.eye(outranking_credibility.shape[0], outranking_credibility.shape[1]))

        def s(x):
            return x * alpha + beta

        while True:
            upper_credibility_th = inherit_credibility_th if inherit_credibility_th != None else np.max(current_outranking_credibility)
            lower_credibility_th = np.max(current_outranking_credibility[current_outranking_credibility < upper_credibility_th - s(upper_credibility_th)], initial=0)

            if upper_credibility_th == 0:
                if dir == 'desc':
                    preorder.append(current_variants)
                else:
                    preorder.insert(0, current_variants)
                break

            cond_credibility = np.zeros_like(current_outranking_credibility)
            cond_select = np.logical_and(current_outranking_credibility > lower_credibility_th, current_outranking_credibility > np.transpose(current_outranking_credibility) + s(current_outranking_credibility))
            cond_credibility[cond_select] = current_outranking_credibility[cond_select]

            strength = np.sum(cond_credibility, axis=1)
            weakness = np.sum(cond_credibility, axis=0)
            quality = strength - weakness

            if dir == 'desc':
                selected_variants = quality == np.max(quality)
            else:
                selected_variants = quality == np.min(quality)

            if current_variants[selected_variants].shape[0] > 1:
                internal_outranking_credibility = current_outranking_credibility[selected_variants, :]
                internal_outranking_credibility = current_outranking_credibility[:, selected_variants]
                internal = self.distilate(
                    internal_outranking_credibility, 
                    current_variants[selected_variants], 
                    dir, 
                    alpha, 
                    beta, 
                    lower_credibility_th
                )
                
                for d in internal:
                    if dir == 'desc':
                        preorder.append(d)
                    else:
                        preorder.insert(0, d)
            else:
                if dir == 'desc':
                    preorder.append(current_variants[selected_variants])
                else:
                    preorder.insert(0, current_variants[selected_variants])

            current_variants = current_variants[~selected_variants]
            current_outranking_credibility = current_outranking_credibility[~selected_variants, :]
            current_outranking_credibility = current_outranking_credibility[:, ~selected_variants]
            

            if current_variants.shape[0] == 0:
                break

        return preorder