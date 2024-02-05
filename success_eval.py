import gurobipy as gp
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
import pickle

import utils
from detection_eval import plant_clique, plant_clique_bipartite

'''
Run manipulation success experiments
'''

PAP_LOAD = 3
REV_LOAD = 6

def max_match(COI, pap_load, rev_load):
    S = np.ones_like(COI)
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    F = m.addMVar(S.shape, lb=0, ub=(1-COI), obj=S)
    m.modelSense = gp.GRB.MAXIMIZE
    m.addConstrs(F[p, :] @ np.ones(S.shape[1]) <= pap_load for p in range(S.shape[0]))
    m.addConstrs(np.ones(S.shape[0]) @ F[:, r] <= rev_load for r in range(S.shape[1]))
    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')
    F_ = F.x
    return F_

def match(S, COI, pap_load, rev_load, verbose=False):
    if verbose:
        stime = time.time()
        print('Constructing LP')

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    F = m.addMVar(S.shape, lb=0, ub=(1-COI), obj=S)
    m.modelSense = gp.GRB.MAXIMIZE
    m.addConstrs(F[p, :] @ np.ones(S.shape[1]) == pap_load for p in range(S.shape[0]))
    m.addConstrs(np.ones(S.shape[0]) @ F[:, r] <= rev_load for r in range(S.shape[1]))

    if verbose:
        print('Done constructing', time.time() - stime)
        stime = time.time()
        print('Solving LP')

    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    if verbose:
        print('Done solving', time.time() - stime)
        stime = time.time()
        print('Outputting LP')   

    F_ = F.x

    if verbose:
        print('Done outputting', time.time() - stime)
    return F_



def success_metrics(trials):
    results = {'frac_total' : [], 'frac_papers' : [], 'frac_revs' : []}
    for F, A, COI in trials:
        F_opt = max_match(COI=COI, pap_load=PAP_LOAD, rev_load=REV_LOAD)
        if F_opt.sum() > 0:
            results['frac_total'].append(F.sum() / F_opt.sum())
        else:
            results['frac_total'].append(1.0)
    
        results['frac_papers'].append((F.sum(axis=1) > 0).mean())
    
        results['frac_revs'].append(((F.T @ A).sum(axis=0) > 0).mean())
    return results

def success_metrics_df(results):
    cols = ['planted_k', 'planted_gamma', 'metric', 'trial', 'value']
    records = []
    for params, trials in results.items():
        agg_results = success_metrics(trials)
        for metric, metric_results in agg_results.items():
            for t, r in enumerate(metric_results):
                records.append((params[0], params[1], metric, t, r))
    return pd.DataFrame.from_records(records, columns=cols)


def plant_clique_attack(BA, B, A, k, gamma, rng):
    b_max = B.max()

    BA, S = plant_clique(BA, k, gamma, rng, A=A)
    assert (B[A == 1] == 0).all()
    assert (A[:, S].sum(0) > 0).all()

    S_mask = np.zeros(BA.shape[0], dtype=bool)
    S_mask[S] = True
    B_mod = B.copy()
    B_mod[:, S_mask] = 0

    # Remove bids on non-colluder eligible papers
    edges_to_filter = (BA > 0)
    edges_to_filter[:, S_mask] = 0
    edges_to_filter[~S_mask, :] = 0
    for (u, v) in np.argwhere(edges_to_filter):
        eligible_papers = np.nonzero((A[:, v] > 0) & (B[:, u] > 0))[0]
        assert eligible_papers.size > 0
        bid_to_keep = rng.choice(eligible_papers)
        B_mod[bid_to_keep, u] = 1
    assert np.all(B_mod <= B)

    # Add bids on colluders
    edges_to_fill = (BA > 0)
    edges_to_fill[:, ~S_mask] = 0
    edges_to_fill[~S_mask, :] = 0
    edges_missing = (BA == 0)
    edges_missing[:, ~S_mask] = 0
    edges_missing[~S_mask, :] = 0
    for (u, v) in np.argwhere(edges_to_fill):
        forbidden_authors_mask = edges_missing[u, :]
        forbidden_papers_mask = (A[:, forbidden_authors_mask].sum(axis=1) > 0)
        eligible_papers = np.nonzero((A[:, v] > 0) & (~forbidden_papers_mask))[0]
        B_mod[eligible_papers, u] = b_max
    B_mod[A == 1] = 0
    assert np.all(B[:, ~S_mask] == B_mod[:, ~S_mask])

    BA_mod = utils.make_bid_auth(B_mod, A)
    assert np.all(BA[~S_mask, :] == BA_mod[~S_mask, :])
    ingroup_mask = np.ix_(S_mask, S_mask)
    assert np.all(BA[ingroup_mask] >= BA_mod[ingroup_mask])
    outgroup_mask = np.ix_(S_mask, ~S_mask)
    assert np.all(BA[outgroup_mask] <= BA_mod[outgroup_mask])

    return BA, B_mod, list(S)

def plant_clique_attack_bipartite(B, A, COI, k, gamma, rng):
    return plant_clique_bipartite(B, A, COI, k, gamma, rng)

def run_experiment(dataset, k, gamma, rng, num_trials, bid_weights, bipartite):
    data_dict = utils.load_data(dataset)
    B = data_dict['bid_matrix']
    A = data_dict['author_matrix']
    BA = utils.make_bid_auth(B=B, A=A)
    COI = data_dict['coi_matrix']
    T = data_dict['tpms_matrix']

    records = []
    for t in range(num_trials):
        t0 = time.time()

        if bipartite:
            B_mod, S = plant_clique_attack_bipartite(B, A, COI, k, gamma, rng)
        else:
            BA_mod, B_mod, S = plant_clique_attack(BA=BA, B=B, A=A, k=k, gamma=gamma, rng=rng)
        if not bid_weights:
            B_mod[B_mod > 0] = 1
        else:
            assert not bipartite

        M_sim = (T * np.power(2, (B_mod / B_mod.max())))
        F = match(S=M_sim, COI=COI, pap_load=PAP_LOAD, rev_load=REV_LOAD)
        t1 = time.time()
        mins = (t1 - t0) / 60

        target_papers = (A[:, S].sum(axis=1) > 0)
        mask = np.ix_(target_papers, S)
        F_sub, A_sub, COI_sub = F[mask], A[mask], COI[mask]
        records.append((F_sub, A_sub, COI_sub))

    return records


def run_experiments_all(dataset, param_list, num_trials, bid_weights, bipartite, verbose=True, save_results=True, dir_name='results', rng_plant=None, suppress_exceptions=True):
    assert dataset in ['aamas_sub3', 'wu']
    assert not bid_weights, 'Should only run without bid weights'

    datestring = datetime.datetime.now().isoformat() 
    fname = f'{dir_name}/success_results_{dataset}' + ('_bipartite' if bipartite else '') + f'_{datestring}'
    results = {}
    print(f'start : {fname}.csv')

    if rng_plant is None:
        rng_plant = np.random.default_rng(seed=0)

    try:
        for gamma, k in param_list:
            if verbose:
                print(f'{dataset} ({k}, {gamma}) : {datetime.datetime.now()}')
            t0 = time.time()
            records = run_experiment(dataset, k, gamma, rng=rng_plant, num_trials=num_trials, bid_weights=bid_weights, bipartite=bipartite)
            t1 = time.time()
            if verbose:
                print(f'{dataset} ({k}, {gamma}) \t ({(t1-t0)/60:.02f})')
            results[(k, gamma)] = records
            if save_results:
                with open(fname + '.pkl', 'wb') as f:
                    pickle.dump(results, f)
                df = success_metrics_df(results)
                df['bid_weights'] = bid_weights 
                df.to_csv(fname + '.csv')
        print(f'finish : {fname}.csv')
    except Exception as e:
        print(f'{repr(e)} : {fname}')
        if not suppress_exceptions:
            raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('-kn', '--k_min', type=int, default=2)
    parser.add_argument('-kx', '--k_max', type=int, default=35)
    parser.add_argument('-gn', '--gamma_min', type=float, default=None)
    parser.add_argument('-gx', '--gamma_max', type=float, default=1)
    parser.add_argument('-gs', '--gamma_step', type=float, default=None)
    parser.add_argument('-t', '--num_trials', type=int, default=10)
    parser.add_argument('-b', '--bid_weights', action='store_true')
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
    run_experiments_all(args.dataset, param_list, args.num_trials, args.bid_weights, args.bipartite, suppress_exceptions=False)

if __name__ == '__main__':
    main()
