import time
from functools import reduce
from itertools import product

import src.discriminate as discri
from src.click_safe import box_click
from src.get_screen import get_screen



useless = set()
def get_around_grids(row, col):
    ret = []
    around = filter(lambda a: a!= (0, 0), product(range(-1, 2), range(-1, 2)))
    for offset in around:
        _col, _row = col + offset[0], row + offset[1]
        if _col < 0 or _col > 29 or _row < 0 or _row > 15:
            continue
        ret.append((_row, _col))
    return ret

def uncertain_grids(matrix, row, col):
    if (row, col) in useless:
        return []
    if not 0 < matrix[row][col] < 9:
        return []
    ret = []
    for _row, _col in get_around_grids(row, col):
        if matrix[_row][_col] == 0: # 寻找未知点，并返回
            ret.append((_row, _col))

    if not ret:
        useless.add((row, col))
    return ret

def is_valid(matrix, row, col):
    '''
    judge if the num of row, col is valid
    '''
    num_mine, num_unkown = 0, 0
    num = matrix[row][col]
    for _row, _col in get_around_grids(row, col):
        if matrix[_row][_col] == -1:
            num_mine += 1
        elif matrix[_row][_col] == 0:
            num_unkown += 1
    if num_mine > num or num_mine + num_unkown < num:
        return False
    return True
def is_around_valid(matrix, row, col):
    for _row, _col in get_around_grids(row, col):
        if not 0 < matrix[_row][_col] < 9:
            continue
        if not is_valid(matrix, _row, _col):
            return False
    return True
def search(matrix, grids, index, result):
    '''
        dfs all available combinations in given grids
    '''
    if index == len(grids):
        l = [1 if matrix[row][col] == -1 else 0 for row, col in grids]
        result.append(l)
        return
    row, col = grids[index]
    # if it is mine
    matrix[row][col] = -1
    if is_around_valid(matrix, row, col):
        search(matrix, grids, index+1, result)
    # matrix[row][col] = 0
    # if it is not mine
    matrix[row][col] = -2
    if is_around_valid(matrix, row, col):
        search(matrix, grids, index+1, result)
    matrix[row][col] = 0



def main():
    start = time.time()


    for k in range (50):
        filename = "./temp/%d.png"%k
        get_screen("difficult",filename)
        matrix = discri.get_matrix("difficult",filename)
        
        row,col = matrix.shape
        mines, safeties = set(), set()
        probability = dict()

        for i in range (row):
            for j in range (col):
                candidates = uncertain_grids(matrix, i, j)
                results = []
                search(matrix, candidates, 0, results)
                stat = reduce(lambda a, b: [x+y for x, y in zip(a,b)], results, [0]*len(candidates))
                # if candidates:
                #     print ((i,j),": \n",candidates,"\n",results,"\n",stat)
                for candidate, num in zip(candidates, stat):
                    if num == 0:
                        safeties.add(candidate)
                    elif num == len(results):
                        mines.add(candidate)
                    elif candidate not in safeties and candidate not in mines:
                        p = num *1.0 / len(results)
                        if candidate in probability:
                            probability[candidate] = max(probability[candidate], p)
                        else:
                            probability[candidate] = p
        
        if not safeties:
            print (probability)
            break

        for row,col in safeties:
            box_click(row,col)

    print ("Use time : ", time.time()-start)

if __name__ == "__main__":
    main()