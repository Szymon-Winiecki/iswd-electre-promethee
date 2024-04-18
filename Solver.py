import numpy as np
import networkx as nx

class Solver():
    def relation_to_preorder(self, relation):
        G = nx.DiGraph()

        groups = {}
        
        for a in range(relation.shape[0]):
            for b in range(relation.shape[1]):
                if a == b:
                    continue
                if relation[a, b] == 0:
                    if a not in groups:
                        groups[a] = set([a])
                    if b not in groups:
                        groups[b] = set([b])

                    groups[a].update(groups[b])
                    groups[b].update(groups[a])

                    for node in groups[a]:
                        groups[node].update(groups[a])


                if relation[a, b] == 1:
                    if not G.has_node(a):
                        G.add_node(a)
                    if not G.has_node(b):
                        G.add_node(b)
                    G.add_edge(a, b)

        edges = [e for e in G.edges]
        for a, b in edges:
            G.remove_edge(a, b)
            if not nx.has_path(G, a, b):
                G.add_edge(a, b)

        for node in groups:
            new_node = groups[node]
            if not G.has_node(new_node):
                nx.relabel_nodes(G, {node: new_node})
            for other in groups[node]:
                if G.has_node(other):
                    G.remove_node(other)

        return G
    
    def preorder_to_ranks(self, preorder):
        roots = [node for node, in_degree in preorder.in_degree() if in_degree == 0]

        ranks = {}

        def go_deeper(node, level):
            if node not in ranks or ranks[node] < level:
                ranks[node] = level
            for neigh in preorder.neighbors(node):
                go_deeper(neigh, level + 1)

        for root in roots:
            go_deeper(root, 1)

        max_depth = max(ranks.values())
        ranks_list = [[] for _ in range(max_depth)]

        for node in ranks:
            ranks_list[ranks[node]-1].append(node)

        return ranks_list