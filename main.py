import numpy as np

from Promethee_I_Solver import Promethee_I_Solver
from Promethee_II_Solver import Promethee_II_Solver
from Electre_III_Solver import Electre_III_Solver

def read_data(path, skip_rows=0, skip_cols=[]):
    data = []
    with open(path, 'r') as file:
        line_no = -1
        for line in file:
            line_no += 1
            if line_no < skip_rows:
                continue
            data.append(line.split())
    
    np_data = np.array(data, dtype=float)

    col_filter = np.full((np_data.shape[1]), True)
    for i in range(col_filter.shape[0]):
        if i in skip_cols:
            col_filter[i] = False

    return np_data[:, col_filter]

def main():

    data = read_data("variants.txt", skip_rows=1, skip_cols=[0])

    # name, type[gain(1) or cost(-1)], weight, q, p, ?veto
    criteria = [
        ['cena',        -1,  3, 50, 150],
        ['powierzchnia', 1,  1, 2, 5],
        ['ocena',        1,  4, 0.2, 0.6],
        ['komunikacja', -1,  2, 50, 150],
        ['centrum',     -1,  4, 3, 10, 20],
        ['komfort',      1,  3, 0, 2, 3],
        ['wystrój',      1,  2, 0, 2],
    ]

    # print(data)

    solver = Promethee_II_Solver(criteria, data)
    solver.solve()

main()