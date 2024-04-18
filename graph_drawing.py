from matplotlib import pyplot as plt
import networkx as nx
import os
import numpy as np

from collections.abc import Iterable

def get_printable_name(n):
    if isinstance(n, np.ndarray):
        return np.array2string(n+1, separator=", ")
    elif isinstance(n, Iterable):
        return '[' + ', '.join([str(x+1) for x in n]) + ']'
    else:
        return '[' + str(n+1) + ']'

def draw_complete_preorder(preorder, directory, filename):
    G = nx.DiGraph()
    
    for rank in range(len(preorder)):
        G.add_node(get_printable_name(preorder[rank]))

        if rank > 0:
            G.add_edge(get_printable_name(preorder[rank-1]), get_printable_name(preorder[rank]))
    
    plt.figure(1,figsize=(12,12)) 
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    plt.cla()
    nx.draw(G, pos=pos, with_labels=True, node_color='white')

    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, filename))

def draw_preorder(preorder_graph, directory, filename):

    nodes_mapping = {node:get_printable_name(node) for node in preorder_graph.nodes}

    G = nx.relabel_nodes(preorder_graph, nodes_mapping, copy=True)

    plt.figure(1,figsize=(12,12)) 
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    plt.cla()
    nx.draw(G, pos=pos, with_labels=True, node_color='white')

    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, filename))

def print_complete_preorder(preorder, directory, filename):

    ranking_str = ""

    for rank in range(len(preorder)):
        ranking_str += f'{rank + 1}. {", ".join([str(x + 1) for x in preorder[rank]])}\n'

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w') as file:
        file.write(ranking_str)
