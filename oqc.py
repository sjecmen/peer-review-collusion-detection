import numpy as np
import pandas as pd
import utils
import time

def edge_count(S, G):
    return G[np.ix_(S, S)].sum()  

def oqc_fs(subsets, edge_counts, alpha):
    return [ec - (alpha * len(S) * (len(S) - 1)) for S, ec in zip(subsets, edge_counts)]

def oqc_f(S, G, alpha):
    s = S.sum()
    return edge_count(S, G) - (alpha * s * (s - 1))



def generate_oqc_local_start_list(G, start_v, rng_init, num_rand_start):
    neighbors = (G[:, start_v] + G[start_v, :])
    neighbors[start_v] += 1
    S_starts = [neighbors > 0]
    for _ in range(num_rand_start):
        v = rng_init.choice(G.shape[0])
        S = np.zeros(G.shape[0], dtype=bool)
        S[v] = 1
        S_starts += [S]
    return S_starts

def oqc_local(G, alpha, start_v, rng_init, max_iter=50, num_rand_start=10):
    f = lambda x : oqc_f(S = x, G=G, alpha=alpha)
    S_starts = generate_oqc_local_start_list(G, start_v, rng_init, num_rand_start)
    trial_results = []
    for S_start in S_starts:
        result = oqc_local_v(f=f, S_start=S_start, num_vertices=G.shape[0], max_iter=max_iter)
        trial_results.append(result)
    S, _ = max(trial_results, key=lambda elt : elt[1])
    return S, f'alpha_{alpha}__max_iter_{max_iter}'

def oqc_local_bipartite(G_mod, alpha, start_v, rng_init, max_iter=50, num_rand_start=10):
    Bids, Poss = G_mod
    Score = Bids - (alpha * Poss)
    def oqc_f_bi(S):
        mask = np.ix_(S, S)
        return Score[mask].sum()

    S_starts = generate_oqc_local_start_list(Bids, start_v, rng_init, num_rand_start)

    trial_results = []
    for trial, S_start in enumerate(S_starts):
        result = oqc_local_v(f=oqc_f_bi, S_start=S_start, num_vertices=Bids.shape[0], max_iter=max_iter)
        trial_results.append(result)
    S, _ = max(trial_results, key=lambda elt : elt[1])
    return S, f'alpha_{alpha}__max_iter_{max_iter}'

def oqc_local_v(f, S_start, num_vertices, max_iter):
    assert S_start.dtype == bool
    S = S_start
    t = 0
    S_prev = None
    while True:
        while True:
            change = False
            for u in range(num_vertices):
                if S[u] == 0:
                    f_current = f(S)
                    S[u] = 1
                    f_new = f(S)
                    if f_new >= f_current:
                        change = True
                    else:
                        S[u] = 0
            if not change:
                break
        change = False
        for u in range(num_vertices):
            if S[u] == 1:
                f_current = f(S)
                S[u] = 0
                f_new = f(S)
                if f_new >= f_current:
                    change = True
                    break
                else:
                    S[u] = 1
        t += 1
        if not change or (max_iter is not None and t >= max_iter) or (S == S_prev).all():
            break
        S_prev = S.copy()
    if f(~S) > f(S):
        S = ~S
    return np.nonzero(S)[0].tolist(), f(S)


def oqc_greedy(G, alpha):
    subsets, edge_counts = greedy_peeling(G)
    fs = oqc_fs(subsets, edge_counts, alpha)
    f_star, S_star = max(zip(fs, subsets))
    return S_star, f'alpha_{alpha}'

def greedy_peeling_frontier(dataset):
    G = utils.make_BA(dataset, authors_only=True)
    subsets, _ = greedy_peeling(G)
    df = pd.DataFrame([(len(S), utils.edge_density(G, S)) for S in subsets], columns=['k', 'edge_density'])
    return df

def greedy_peeling(G, verbose=False):
    assert all([G[i, i] == 0 for i in range(G.shape[0])])
    indegrees = G.sum(axis=0)
    outdegrees = G.sum(axis=1)
    degrees = indegrees + outdegrees
    degree_map = {d : [] for d in range(int(degrees.max()) + 1)}
    for i in range(G.shape[0]):
        degree_map[degrees[i]].append(i)
    adjacency_map = {i : [] for i in range(G.shape[0])}
    for (i, j) in np.argwhere(G):
        adjacency_map[i].append(j)
        adjacency_map[j].append(i)
    # Find order of vertex removal
    order = []
    min_degree = 0
    while len(order) < G.shape[0]:
        while len(degree_map[min_degree]) == 0:
            min_degree += 1
        v = degree_map[min_degree].pop()
        degrees[v] = -1
        order.append(v)
        neighbors = adjacency_map[v]
        for u in neighbors:
            if degrees[u] == -1:
                continue
            degree_map[degrees[u]].remove(u)
            degrees[u] -= 1
            degree_map[degrees[u]].append(u)
            if degrees[u] < min_degree:
                min_degree = degrees[u]
    # Resulting subsets are V, V-{order[0]}, V-{order[0:i]}, etc
    subsets = []
    current_subset = []
    edge_counts = []
    current_edge_count = 0
    for v in reversed(order):
        # Add number of edges from v to current_subset
        current_edge_count += G[current_subset, v].sum() + G[v, current_subset].sum() 
        current_subset.append(v)
        subsets.append(current_subset.copy())
        edge_counts.append(current_edge_count)
    return subsets, edge_counts

def greedy_peeling_frontier_bipartite(dataset):
    matrices = utils.get_author_only_matrices(dataset)
    B = matrices['bid_matrix']
    A = matrices['author_matrix']
    COI = matrices['coi_matrix']
    subsets, densities = greedy_peeling_bipartite_reviewers(B, A, COI)
    df = pd.DataFrame([(len(S), d) for (S, d) in zip(subsets, densities)], columns=['k', 'bid_density'])
    return df

def greedy_peeling_bipartite_reviewers(B, A, COI):
    BPs = utils.get_BP_matrices(B, A, COI)
    current_S = list(range(B.shape[1]))
    d = utils.edge_density_bipartite_BP(*BPs, current_S)
    subsets = [current_S]
    densities = [d]
    while len(current_S) > 0:
        candidate_subsets = []
        for r in current_S:
            t0 = time.time()
            candidate_S = current_S.copy() 
            t1 = time.time()
            candidate_S.remove(r)
            t2 = time.time()
            d = utils.edge_density_bipartite_BP(*BPs, current_S)
            t3 = time.time()
            candidate_subsets.append((candidate_S, d))
        S, d = max(candidate_subsets, key=lambda p : p[1])
        subsets.append(S)
        densities.append(d)
        current_S = S
    return subsets, densities



def oqc_adaptive_alpha(G, target_size):
    k = target_size
    subsets, edge_counts = greedy_peeling(G)
    S_star = subsets[k-1]
    ec_star = edge_counts[k-1]
    assert len(S_star) == k, len(S_star)
    S_plus = subsets[k]
    ec_plus = edge_counts[k]
    assert len(S_plus) == k+1
    S_minus = subsets[k-2]
    ec_minus = edge_counts[k-2]
    assert len(S_minus) == k-1
    alpha_lb = (ec_plus - ec_star) / (2 * k)
    alpha_ub = (ec_star - ec_minus) / (2 * (k-1))
    if alpha_ub >= alpha_lb:
        alpha = (alpha_lb + alpha_ub) / 2
    else:
        # select alpha to be generous (allow returning of larger group)
        alpha = alpha_ub # smaller value so larger group favored, S_k = S_{k-1} <= S_{k+1}
        # local optimum will be at the minimum k' st k' > k
    fs = oqc_fs(subsets, edge_counts, alpha)
    f_star, S_star = max(zip(fs, subsets))
    return S_star, alpha

def oqc_greedy_adaptive(G, target_size):
    S_star, alpha = oqc_adaptive_alpha(G, target_size)
    return S_star, f'alpha_{alpha}'

def oqc_local_adaptive(G, target_size, start_v, rng_init, max_iter=50, num_rand_start=10):
    _, alpha = oqc_adaptive_alpha(G, target_size)
    return oqc_local(G, alpha, start_v, rng_init, max_iter, num_rand_start)


