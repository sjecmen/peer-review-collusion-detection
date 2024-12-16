import gurobipy as gp
import numpy as np
import torch
import math
import pandas as pd
import itertools
import os

def load_data(dataset):
    if dataset == 'wu_variant':
        data_dict = load_data_wu(True)
    elif dataset == 'aamas_sub3_variant':
        data_dict = load_data_aamas_sub3(True)
    elif dataset == 'wu':
        data_dict = load_data_wu(False)
    elif dataset == 'aamas_sub3':
        data_dict = load_data_aamas_sub3(False)
    else:
        assert False
    return data_dict

def load_data_wu(variant=False):
    '''
    Dataset sourced from (Wu et al., 2021)
    '''
    n_rev, n_pap = 2483, 2446
    shape = (n_pap, n_rev)
    tensor_data = torch.load("datasets/wu_tensor_data.pl")
    
    TPMS = tensor_data['tpms'].numpy().T
    assert TPMS.shape == shape
    
    B = tensor_data['label'].T
    assert B.shape == shape

    def dict_to_mat(dict_sp_t):
        a = torch.sparse.FloatTensor(dict_sp_t["indices"], dict_sp_t["values"], dict_sp_t["size"])
        b = a.to_dense().numpy()
        return b
    r_subject = dict_to_mat(tensor_data['r_subject'])
    assert r_subject.shape[0] == n_rev

    p_subject = dict_to_mat(tensor_data['p_subject'])
    assert p_subject.shape[0] == n_pap

    A = np.load('datasets/wu_authorship.npy')
    assert A.shape == shape

    if variant:
        rng = np.random.default_rng(seed=2)
        B_orig = B.copy()
        idx = rng.choice(np.argwhere(B > 0), size=15000, replace=False) # > 10%
        idx2 = rng.choice(np.argwhere(B == 0), size=15000, replace=False)
        B[tuple(idx.T)] = 0
        B[tuple(idx2.T)] = 1

    B[A == 1] = 0
    B = B.astype(np.float32)
    A = A.astype(np.float32)

    return {'n_rev' : n_rev,
            'n_pap' : n_pap,
            'bid_matrix' : B,
            'tpms_matrix' : TPMS,
            'rev_areas' : r_subject,
            'pap_areas' : p_subject,
            'author_matrix' : A,
            'coi_matrix' : A.copy()}

def load_data_aamas():
    '''
    Dataset sourced from PrefLib
    '''
    aamas = pd.read_csv('datasets/aamas_2021.csv')
    aamas = aamas[~aamas['Bidder'].str.contains('spc')]
    aamas['rid'] = aamas['Bidder'].str.slice_replace(0, 3).astype(int)
    aamas['pid'] = aamas['Submission'].astype(int)
    aamas = aamas.pivot(columns='rid', index='pid', values='Bid')
    B = aamas.replace({'yes' : 2, 'maybe' : 1, 'conflict' : 0, np.nan : 0}).to_numpy()
    A = aamas.replace({'yes' : 0, 'maybe' : 0, 'conflict' : 1, np.nan : 0}).to_numpy()
    assert np.all(B[A == 1] == 0)
    B = B.astype(np.float32)
    A = A.astype(np.float32)

    data = {'n_rev' : B.shape[1], 
            'n_pap' : B.shape[0],
            'bid_matrix' : B,
            'author_matrix' : A,
            'coi_matrix' : A.copy()}
    assert os.path.exists('datasets/aamas_text.npy')
    T = np.load('datasets/aamas_text.npy')
    data['tpms_matrix'] = T
    return data

def load_data_aamas_sub3(variant=False):
    # AAMAS dataset, subsampling 3 authors UAR
    data = load_data_aamas()
    A_sub = np.load('datasets/aamas_authorship.npy')
    if variant:
        A_sub = np.load('datasets/aamas_authorship2.npy')
    data['author_matrix'] = A_sub
    return data

def make_BA(dataset, return_text_weights=False, authors_only=False):
    data_dict = load_data(dataset)
    B = data_dict['bid_matrix']
    A = data_dict['author_matrix']
    if authors_only:
        authors = (np.sum(A, axis=0) > 0)
        B = B[:, authors]
        A = A[:, authors]
    BA = make_bid_auth(B=B, A=A)
    if return_text_weights:
        if 'tpms_matrix' in data_dict:
            TPMS = data_dict['tpms_matrix']
            if authors_only:
                TPMS = TPMS[:, authors]
            T, W = make_text_weights(B=B, A=A, TPMS=TPMS)
            assert T.shape == BA.shape
        else:
            T, W = None, None
        return BA, T, W
    else:
        return BA

def make_bid_auth(B, A):
    BA = ((B.T @ A) > 0).astype(np.float32)
    assert np.all(np.diag(BA) == 0)
    return BA

def get_author_only_matrices(dataset):
    data_dict = load_data(dataset)
    authors = (np.sum(data_dict['author_matrix'], axis=0) > 0)
    return {n : data_dict[n][:, authors] for n in ['bid_matrix', 'author_matrix', 'coi_matrix', 'tpms_matrix']}


def make_text_weights(B, A, TPMS):
    # Return two matrices: T=total text similarity for (r1, r2) bids, W=total number of (r1, r2) bids 
    B_ind = (B > 0).astype(int)
    A_ind = (A > 0).astype(int)
    T = (TPMS * B_ind).T @ A_ind
    W = B_ind.T @ A_ind
    return T, W

def edge_density(G, S):
    k = len(S)
    if k < 2:
        return 0
    max_edges = 2 * math.comb(k, 2)
    edges = G[np.ix_(S, S)].sum()
    assert np.all((G == 0) | (G == 1))
    assert edges <= max_edges, f'{edges} {max_edges}'
    return edges / max_edges

def get_BP_matrices(B, A, COI):
    Bids = (B > 0).astype(float).T @ (A > 0).astype(float)
    Poss = (COI == 0).astype(float).T @ (A > 0).astype(float)
    return Bids, Poss

def edge_density_bipartite_BP(Bids, Poss, S):
    mesh = np.ix_(S, S)
    total_poss = Poss[mesh].sum()
    return (Bids[mesh].sum() / total_poss) if total_poss > 0 else 0

def edge_density_bipartite(B, A, COI, S):
    total_bids = 0
    total_poss = 0
    for i, j in itertools.permutations(S, 2):
        total_bids += ((B[:, i] > 0) & (A[:, j] > 0)).sum()
        total_poss += ((COI[:, i] == 0) & (A[:, j] > 0)).sum()
    return (total_bids / total_poss) if total_poss > 0 else 0

def make_param_list(k_min, k_max, gamma_min, gamma_max, gamma_step, k_step=1):
    ks = np.arange(k_min, k_max + 1, k_step).tolist()
    num_gamma = int(np.round((gamma_max - gamma_min) / gamma_step) + 1) if gamma_step > 0 else 1
    gammas = np.linspace(gamma_max, gamma_min, num_gamma).round(decimals=1).tolist()
    param_list = itertools.product(gammas, ks)
    return param_list

if __name__ == "__main__":
    load_data_aamas_sub3(variant=True)
    load_data_wu(variant=True)
