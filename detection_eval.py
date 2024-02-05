import numpy as np
import time
import sys
import math
import pandas as pd
import functools
import random
import itertools
import datetime
import os 
import argparse

import utils
import ds
import oqc
import telltail
import fraudar

def get_local_search_init(BA_mod):
    # Return a heuristic reviewer to start at based on triangle-counting heuristic
    S_starts = []
    G = BA_mod # BA
    degs = G.sum(axis=0) + G.sum(axis=1)
    triangles = np.diag(G @ G @ G)
    heuristic = np.divide(triangles, degs, out=np.zeros(degs.shape), where=(degs > 0)).argmax()
    return heuristic

def get_local_search_init_bipartite(B_mod, A):
    # Return a heuristic reviewer to start at based on 4-cycle-count heuristic
    S_starts = []
    B = B_mod
    degs = B.sum(axis=0) + A.sum(axis=0)
    t0 = time.time()
    BA = (B.T @ A)
    butterflies = np.diag(BA @ BA)
    t1 = time.time()
    heuristic = np.divide(butterflies, degs, out=np.zeros(degs.shape), where=(degs > 0)).argmax()
    assert heuristic < B.shape[1]
    return heuristic

def plant_clique(BA, k, gamma, rng, A):
    BA = BA.copy()

    authors = (np.sum(A, axis=0) > 0)
    S = rng.choice(np.arange(BA.shape[0])[authors], size=k, replace=False)
    assert authors[S].all()

    mask = np.ix_(S, S)
    num_to_add = int(np.ceil(gamma * (k * (k - 1))) - np.sum(BA[mask]))
    if num_to_add > 0:
        missing_edge_mask = np.zeros(BA.shape, dtype=bool)
        missing_edge_mask[mask] = (BA[mask] == 0)
        missing_edge_mask[np.eye(BA.shape[0]) == 1] = 0
        missing_edges = np.argwhere(missing_edge_mask)
        assert missing_edges.size > 0
        rng.shuffle(missing_edges, axis=0)
        edges_to_add = missing_edges[:num_to_add]
        assert edges_to_add.shape == (num_to_add, 2), f'{edges_to_add.shape}, {num_to_add}'
        BA[tuple(zip(*edges_to_add))] = 1
    assert utils.edge_density(BA, S) >= gamma, utils.edge_density(BA, S)
    assert np.diag(BA).all() == 0
    return BA, list(S)

def plant_clique_bipartite(B, A, COI, k, gamma, rng):
    B = B.copy()
    if rng is None:
        rng = np.random.default_rng()

    authors = (np.sum(A, axis=0) > 0)
    S = rng.choice(np.arange(B.shape[1])[authors], size=k, replace=False)
    assert authors[S].all()

    Bids, Poss = utils.get_BP_matrices(B, A, COI)
    mesh = np.ix_(S, S)
    total_poss = Poss[mesh].sum()
    total_bids = Bids[mesh].sum()
    target_num_bids = np.ceil(total_poss * gamma)
    num_to_add = int(target_num_bids - total_bids)

    if num_to_add > 0:
        missing_bids = []
        for i, j in itertools.permutations(S, 2):
            missing_bids_i = np.nonzero((B[:, i] == 0) & (A[:, j] > 0) & (COI[:, i] == 0))[0]
            missing_bids += [(p, i) for p in missing_bids_i]
        assert len(missing_bids) == total_poss - total_bids
        rng.shuffle(missing_bids)
        bids_to_add = missing_bids[:num_to_add]
        B[tuple(zip(*bids_to_add))] = 1
    assert utils.edge_density_bipartite(B, A, COI, S) >= gamma
    assert np.all(B[A == 1] == 0) and np.all(B[COI == 1] == 0)
    return B, list(S)

method_fn_dict = {
    'densest_subgraph' : ds.densest_subgraph,
    'oqc_greedy' : functools.partial(oqc.oqc_greedy, alpha=(1/3)),
    'oqc_local' : functools.partial(oqc.oqc_local, alpha=(1/3)),
    'oqc_greedy_adaptive' : oqc.oqc_greedy_adaptive,
    'oqc_local_adaptive' : oqc.oqc_local_adaptive,
    'telltail' : telltail.run_telltail,
    'oqc_local_bipartite' : functools.partial(oqc.oqc_local_bipartite, alpha=(1/3)),
    'fraudar' : fraudar.run_fraudar,
    'oqc_local_heuristic-start' : functools.partial(oqc.oqc_local, alpha=(1/3), num_rand_start=0),
    'oqc_local_adaptive_heuristic-start' : functools.partial(oqc.oqc_local_adaptive, num_rand_start=0),
    'oqc_local_bipartite_heuristic-start' : functools.partial(oqc.oqc_local_bipartite, alpha=(1/3), num_rand_start=0),
}

def get_method_fn(method):
    return method_fn_dict[method]

def bid_adjacency_matrix(B):
    # Make a symmetric adjacency matrix from the bids
    n = B.shape[0] + B.shape[1]
    G = np.zeros((n, n), dtype=np.float32)
    G[:B.shape[0], B.shape[0]:] = B
    G[B.shape[0]:, :B.shape[0]] = B.T
    G[G > 0] = 1
    assert np.all((G == 0) | (G == 1))
    assert np.all(G == G.T)
    return G

def bid_adjacency_matrix_asym(B):
    # Make an asymmetric adjacency matrix from the bids
    B = B.copy().astype(np.float32)
    G = B.T
    G[G > 0] = 1
    assert np.all((G == 0) | (G == 1))
    return G

def run_experiment(matrices, k, gamma, method, graph_type, bipartite, num_trials, rng_plant, rng_init):
    # Run one method, with the given graph type
    npap, nrev = matrices['B'].shape

    records = []
    for t in range(num_trials):
        shift_output = False
        if bipartite:
            B_mod, S_true = plant_clique_bipartite(matrices['B'], matrices['A'], matrices['COI'], k, gamma, rng_plant)
            actual_density = utils.edge_density_bipartite(B_mod, matrices['A'], matrices['COI'], S_true)

            if method == 'fraudar':
                if graph_type == 'B':
                    G_mod = bid_adjacency_matrix_asym(B_mod)
                elif graph_type == 'BuA':
                    G_mod = bid_adjacency_matrix_asym(B_mod + matrices['A'])
                else:
                    assert False, 'graph type not specified'
            elif 'oqc_local_bipartite' in method:
                G_mod = utils.get_BP_matrices(B_mod, matrices['A'], matrices['COI'])
            else:
                if graph_type == 'B':
                    G_mod = bid_adjacency_matrix(B_mod)
                    shift_output = True
                elif graph_type == 'BuA':
                    G_mod = bid_adjacency_matrix(B_mod + matrices['A'])
                    shift_output = True
                else:
                    assert False, 'graph type not specified'
        else:
            assert method not in ['fraudar', 'oqc_local_bipartite']
            assert graph_type is None
            BA_mod, S_true = plant_clique(matrices['BA'], k, gamma, rng=rng_plant, A=matrices['A'])
            actual_density = utils.edge_density(BA_mod, S_true)
            G_mod = BA_mod
            assert G_mod.dtype == np.float32

        extra_args = {}
        if 'oqc_local' in method or 'telltail' == method:
            if bipartite:
                extra_args['start_v'] = get_local_search_init_bipartite(B_mod=B_mod, A=matrices['A']) + (npap if shift_output else 0)
            else:
                extra_args['start_v'] = get_local_search_init(BA_mod=BA_mod) + (npap if shift_output else 0)
            extra_args['rng_init'] = rng_init
        if 'adaptive' in method:
            if shift_output:
                extra_args['target_size'] = (matrices['A'][:, S_true].sum(axis=1) > 0).sum() + k
            else:
                extra_args['target_size'] = k

        method_fn = get_method_fn(method)
        t0 = time.time()
        S_detect, method_param_string = method_fn(G_mod, **extra_args)
        t1 = time.time()
        mins = (t1 - t0) / 60

        
        if shift_output:
            assert not all([s < npap for s in S_detect])
            S_detect = [s - npap for s in S_detect if s >= npap]
        num_detected = len(S_detect)
        tp = np.intersect1d(S_true, S_detect).size
        if bipartite:
            detected_density = utils.edge_density_bipartite(B_mod, matrices['A'], matrices['COI'], S_detect)
        else:
            detected_density = utils.edge_density(BA_mod, S_detect)
        records.append((method, graph_type, k, gamma, bipartite,
            t, tp, num_detected, nrev, mins, 
            method_param_string, detected_density, actual_density))
    df = pd.DataFrame.from_records(records, columns=['method', 'graph_type', 'planted_k', 'planted_gamma', 'bipartite', 
        'trial', 'num_true_positive', 'num_detected', 'num_total', 'time', 
        'method_params', 'detected_density', 'actual_density'])
    return df



def run_experiments_all(dataset, param_list, method, num_trials, graph_type, bipartite, verbose=True, save_results=True, rng_plant=None, rng_init=None, dir_name='results', suppress_exceptions=True):
    assert dataset in ['aamas_sub3', 'wu']
    assert method in method_fn_dict
    assert graph_type in ['B', 'BuA', None]

    data_dict = utils.load_data(dataset)
    B = data_dict['bid_matrix']
    A = data_dict['author_matrix']
    BA = utils.make_bid_auth(B=B, A=A)
    matrices = {'B' : B, 'A' : A, 'BA' : BA, 'COI' : data_dict['coi_matrix']}

    datestring = datetime.datetime.now().isoformat() 
    fname = f'{dir_name}/detection_results_{dataset}_{method}' + (f'_{graph_type}' if graph_type is not None else '') + ('_bipartite' if bipartite else '') + f'_{datestring}.csv'

    print(f'start : {fname}')
    results = []
    if rng_plant is None:
        rng_plant = np.random.default_rng(seed=0)
    if rng_init is None:
        rng_init = np.random.default_rng(seed=1)
    try:
        for gamma, k in param_list:
            if verbose:
                print(f'{dataset} {method}-{graph_type} ({k}, {gamma}) : {datetime.datetime.now()}')
            t0 = time.time()
            df = run_experiment(matrices=matrices, k=k, gamma=gamma, method=method, graph_type=graph_type, bipartite=bipartite,
                    num_trials=num_trials, rng_plant=rng_plant, rng_init=rng_init)
            t1 = time.time()
            jac = (df['num_true_positive'] / (df['num_detected'] + df['planted_k'] - df['num_true_positive'])).mean()
            if verbose:
                print(f'{dataset} {method}-{graph_type} ({k}, {gamma}) \t {jac:.02f} \t ({(t1-t0)/60:.02f})')
            results.append(df)
            if save_results:
                pd.concat(results).to_csv(fname)
        print(f'finish : {fname}')
    except Exception as e:
        print(f'{repr(e)} : {fname} ')
        if not suppress_exceptions:
            raise e


def run_peeling(dataset, bipartite):
    if bipartite:
        df = oqc.greedy_peeling_frontier_bipartite(dataset)
    else:
        df = oqc.greedy_peeling_frontier(dataset)
    df.to_csv(f'results/peeling_results_{dataset}' + ('_bipartite' if bipartite else '') + '.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('-m', '--method', default=None)
    parser.add_argument('-kn', '--k_min', type=int, default=2)
    parser.add_argument('-kx', '--k_max', type=int, default=35)
    parser.add_argument('-gn', '--gamma_min', type=float, default=None)
    parser.add_argument('-gx', '--gamma_max', type=float, default=1)
    parser.add_argument('-gs', '--gamma_step', type=float, default=None)
    parser.add_argument('-t', '--num_trials', type=int, default=10)
    parser.add_argument('-f', '--frontier', action='store_true', help='Run greedy peeling frontier')
    parser.add_argument('-g', '--graph_type', default=None, help="For some bipartite detection algorithms, which edge set to use. Values are either `B` or `BuA`.")
    parser.add_argument('-bp', '--bipartite', action='store_true')
    parser.add_argument('-z', '--test', action='store_true')

    args = parser.parse_args()
    if args.gamma_step is None:
        args.gamma_step = 0.2 if args.bipartite else 0.1
    if args.gamma_min is None:
        args.gamma_min = 0.2 if args.bipartite else 0.5

    if args.test:
        print('test run')
        ks = [5, 10, 15, 20, 25]
        if args.bipartite:
            gammas = [1, 0.6, 0.2]
        else:
            gammas = [1, 0.8, 0.6]
        param_list = list(itertools.product(gammas, ks))
    else:
        param_list = utils.make_param_list(args.k_min, args.k_max, args.gamma_min, args.gamma_max, args.gamma_step)

    if args.frontier:
        run_peeling(args.dataset, args.bipartite)
    else:
        run_experiments_all(args.dataset, param_list, args.method, args.num_trials, args.graph_type, args.bipartite, suppress_exceptions=False)

if __name__ == '__main__':
    main()

