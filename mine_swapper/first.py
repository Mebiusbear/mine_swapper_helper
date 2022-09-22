import numpy as np
mines, safeties = set(), set()
probability = {}
matrix = np.array()
useless = set()

from itertools import product
around = filter(lambda a: a!= (0, 0), product(range(-1, 2), range(-1, 2)))
def get_around_grids(col, row):
    ret = []
    for offset in around:
        _col, _row = col + offset[0], row + offset[1]
        if _col < 0 or _col >= W or _row < 0 or _row >= H:
            continue
        ret.append((_col, _row))
    return ret

def uncertain_grids(matrix, col, row):
    
    if (col, row) in useless:
        return []

    if not 0 < matrix[col][row] < 9:
        return []

    ret = []
    for _col, _row in get_around_grids(col, row):
        if matrix[_col][_row] == 0: # 寻找未知点，并返回
            ret.append((_col, _row))

    if not ret:
        useless.add((col, row))
    return ret

def search(matrix, grids, index, result):
    '''
        dfs all available combinations in given grids
    '''
    if index == len(grids):
        l = [1 if matrix[col][row] == '*' else 0 for col, row in grids]
        result.append(l)
        return

    col, row = grids[index]
    # if it is mine
    matrix[col][row] = '*'
    if is_around_valid(matrix, col, row):
        search(matrix, grids, index+1, result)
    matrix[col][row] = 0
    # if it is not mine
    matrix[col][row] = ' '
    if is_around_valid(matrix, col, row):
        search(matrix, grids, index+1, result)
    matrix[col][row] = 0



h,w = matrix.shape
for col in range (w):
    for row in  range (h):
        candidates = uncertain_grids(matrix, col, row)
        if not candidates:
            continue
        print ('search: ', col, row, candidates)
        results = []
        search(matrix, candidates, 0, results)