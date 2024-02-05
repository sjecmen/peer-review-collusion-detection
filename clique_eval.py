import math
import numpy as np
import itertools
import subprocess
import sys
import utils
import time
import pandas as pd
import datetime
import argparse
import os

def save_graph(matrices, k, gamma, fname):
    BA = matrices['BA']
    n = BA.shape[0]
    with open(fname, 'w') as f:
        f.write(f'{n}\n')
        f.write(f'{k}\n')
        f.write(f'{gamma:.02f}\n')
        for i, j in itertools.product(range(n), range(n)):
            if BA[i, j] == 1:
                f.write(f'{i} {j}\n')

def save_graph_bipartite(matrices, k, gamma, fname):
    B = matrices['bid_matrix']
    A = matrices['author_matrix']
    COI = matrices['coi_matrix']
    n = B.shape[1]
    with open(fname, 'w') as f:
        f.write(f'{n}\n')
        f.write(f'{k}\n')
        f.write(f'{gamma:.02f}\n')
        for i, j in itertools.product(range(n), range(n)):
            b = ((B[:, i] > 0) & (A[:, j] > 0)).sum()
            p = ((COI[:, i] == 0) & (A[:, j] > 0)).sum()
            f.write(f'{i} {j} {b} {p}\n')

def count_cliques_c(matrices, k, gamma, dataset, timeout, bipartite):
    time_tag = time.time_ns()
    if bipartite:
        fname = f'_graph_{dataset}_bipartite_{time_tag}.txt'
        exename = f"./count_cliques_bipartite.out"
        save_graph_bipartite(matrices, k, gamma, fname)
    else:
        fname = f'_graph_{dataset}_{time_tag}.txt'
        exename = f"./count_cliques_c.out"
        save_graph(matrices, k, gamma, fname)

    timeout_flag = False
    try:
        result = subprocess.run([exename, fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=(timeout*60))
        output = result.stdout.decode('utf-8').strip()
        lines = output.split('\n')
        x = int(lines[0])
        hist = [int(y) for y in lines[1:]]
    except subprocess.TimeoutExpired as e:
        print(f'Timeout({e.cmd})')
        timeout_flag = True

        if e.stderr is None:
            lines = []
        else:
            lines = e.stderr.decode('utf-8').strip().split('\n')
        if len(lines) == 0:
            x = 0 
            hist = []
        else:
            if bipartite:
                hist = [int(y) for y in lines[-1].strip().split(' ')]
                if len(hist) < 11:
                    print(hist)
                    hist = [int(y) for y in lines[-2].strip().split(' ')]
                    print(hist)
                assert len(hist) == 11
                x = sum(hist)
            else:
                x = int(lines[-1])
                hist = []
    os.remove(fname)
    return x, hist, timeout_flag



def run(dataset, param_list, timeout, bipartite, dir_name='results', verbose=True):
    frontier = False
    if param_list == 'frontier':
        frontier = True 
        param_list = [(1.0, 2)]

    if bipartite:
        matrices = utils.get_author_only_matrices(dataset)
    else:
        matrices = {'BA' : utils.make_BA(dataset, return_text_weights=False, authors_only=True)}

    datestring = datetime.datetime.now().isoformat() 
    fname = f'{dir_name}/clique_results_{dataset}' + ('_bipartite' if bipartite else '') + f'_{datestring}.csv'
    print(fname)
    i = 0
    param_list = list(param_list)
    results = []
    infeasible_set = set()
    while i < len(param_list):
        gamma, k = param_list[i]
        if any([gamma <= g_inf and k >= k_inf for (g_inf, k_inf) in infeasible_set]):
            if verbose:
                print(f'({k}, {gamma}) : skipping due to known infeasibility')
            i += 1
            continue
        else:
            if verbose:
                print(f'({k}, {gamma}) : {datetime.datetime.now()}')
        t0 = time.time()
        c, hist, timeout_flag = count_cliques_c(matrices, k, gamma, dataset, timeout=timeout, bipartite=bipartite)
        t1 = time.time()
        mins = (t1-t0)/60
        if bipartite:
            assert timeout_flag or (len(hist) == 11 and sum(hist) == c)
            sums = np.cumsum(hist[::-1])[::-1]
            for bucket, (gamma_count, cumulative_count) in enumerate(zip(hist, sums)):
                g = bucket / 10
                results.append((k, g, cumulative_count, dataset, mins, timeout_flag))
            for j, count in enumerate(sums[1:]):
                g = (j+1) / 10
                if verbose:
                    if timeout_flag:
                        print(f'({k}, {g}) \t ({count}) \t ({mins:.02f})')
                    else:
                        print(f'({k}, {g}) \t {count} \t ({mins:.02f})')
        else:
            assert len(hist) == 0
            if verbose:
                if timeout_flag:
                    print(f'({k}, {gamma}) \t ({c}) \t ({mins:.02f})')
                else:
                    print(f'({k}, {gamma}) \t {c} \t ({mins:.02f})')
            results.append((k, gamma, c, dataset, mins, timeout_flag))
        df = pd.DataFrame.from_records(results, columns=['k', 'gamma', 'num_cliques', 'dataset', 'time', 'timeout_flag'])
        df.to_csv(fname)

        if frontier:
            if timeout or (gamma == 0.1):
                return df
            elif c == 0:
                param_list.append((np.round(gamma - 0.1, 1), k))
            else:
                param_list.append((gamma, k + 1))
        if c == -1:
            infeasible_set.add((gamma, k))
        i += 1
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('-t', '--time_limit', type=int, default=1440, help='Time limit in minutes')
    parser.add_argument('-kn', '--k_min', type=int, default=2)
    parser.add_argument('-kx', '--k_max', type=int, default=None)
    parser.add_argument('-gn', '--gamma_min', type=float, default=0.5)
    parser.add_argument('-gx', '--gamma_max', type=float, default=1)
    parser.add_argument('-gs', '--gamma_step', type=float, default=0.1)
    parser.add_argument('-bp', '--bipartite', action='store_true')
    parser.add_argument('-f', '--frontier', action='store_true', help='Trace the frontier of feasible params')

    args = parser.parse_args()
    assert args.dataset in ['aamas_sub3', 'wu']
    if args.bipartite:
        args.gamma_min = 1
        args.gamma_max = 1
        args.gamma_step = 0
    if args.k_max is None:
        if args.bipartite:
            args.k_max = 7
        else:
            args.k_max = 11
    
    if args.frontier:
        param_list = 'frontier'
    else:
        param_list = utils.make_param_list(args.k_min, args.k_max, args.gamma_min, args.gamma_max, args.gamma_step)
    run(args.dataset, param_list, args.time_limit, args.bipartite)


if __name__ == '__main__':
    main()
