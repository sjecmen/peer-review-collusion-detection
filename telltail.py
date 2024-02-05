import numpy as np

'''
TellTail code is adapted from (Hooi et al., 2020)
'''

def generate_telltail_start_list(G, start_v, rng_init):
    neighbors = (G[:, start_v] + G[start_v, :])
    neighbors[start_v] += 1
    S_starts = [neighbors > 0] + [rng_init.random(G.shape[0]) > rng_init.random(1) for _ in range(10)]
    return S_starts

def run_telltail(G, start_v, rng_init):
    if np.any(G != G.T):
        G_undir = np.clip(G * G.T, 0, 1) # intersection
        graph_type = 'intersection'
    else:
        G_undir = G
        graph_type = 'none'

    S_starts = generate_telltail_start_list(G, start_v, rng_init)
    trial_results = []
    for S_start in S_starts:
        result = detect_telltail(G_undir, S_start)
        trial_results.append(result)
    x_star, f_star = max(trial_results, key=lambda elt : elt[1])
    S_detect = list(np.nonzero(x_star)[0])
    method_param_string = f'proj_type_{graph_type}'
    return S_detect, method_param_string

def modularity_matrix(A):
    d = np.sum(A, axis=0)
    m = np.sum(d) / 2
    M = A - np.outer(d, d) / (2 * m)
    return M

def detect_telltail(A, S_start):
    n = A.shape[0]
    M = modularity_matrix(A)

    x = S_start #rng.random(n) > rng.random(1)
    deg = np.sum(M[x, :], axis=0).reshape(-1, 1)
    score = metric_tail3(A, x)

    while True:
        deg_add = deg.copy()
        deg_add[x] = np.nan
        deg_del = deg.copy()
        deg_del[~x] = np.nan

        try:
            idx_add = np.nanargmax(deg_add)
        except ValueError:
            idx_add = None
        try:
            idx_del = np.nanargmin(deg_del)
        except ValueError:
            idx_del = None

        #print(f'deg_add={np.nanmax(deg_add):.1f}, deg_del={np.nanmin(deg_del):.1f}')

        x_add = x.copy()
        if idx_add is not None:
            x_add[idx_add] = 1
        x_del = x.copy()
        if idx_del is not None:
            x_del[idx_del] = 0

        score_add = metric_tail3(A, x_add)
        score_del = metric_tail3(A, x_del)

        if np.sum(x) == 0:
            assert idx_del is None
            score_del = -np.inf
        if np.sum(x) == n:
            assert idx_add is None
            score_add = -np.inf

        #print(f'size={np.sum(x)}, edges={(np.dot(x.T, np.dot(A, x)) / 2)[0, 0]:.0f}, score={score:.3f}, score_add={score_add:.3f}, score_del={score_del:.3f}', end=' ')

        if score >= score_add and score >= score_del:
            #print('-> local opt')
            break
        elif score_add >= score_del:
            #print('-> add')
            deg = deg + M[:, idx_add].reshape(-1, 1)
            x = x_add
            score = score_add
        else:
            #print('-> del')
            deg = deg - M[:, idx_del].reshape(-1, 1)
            x = x_del
            score = score_del

    return x, score

def metric_tail3(A, x):
    n = A.shape[0]
    k = np.sum(x)

    if k == 0 or k == n:
        score = 0
    else:
        s = np.floor(n / 2).astype(int)
        deg = np.sum(A, axis=0)
        m = np.sum(deg) / 2

        sumB = np.sum(deg**2) / (4 * m)
        sumB2 = m + (np.sum(deg**2)**2 - np.sum(deg**4)) / (8 * m**2) - np.dot(deg, np.dot(A, deg)) / (2 * m)
        sumBrow2 = np.sum((deg**2 / (2 * m))**2)

        p2 = s * (s - 1) / (n * (n - 1))
        p3 = s * (s - 1) * (s - 2) / (n * (n - 1) * (n - 2))
        p4 = s * (s - 1) * (s - 2) * (s - 3) / (n * (n - 1) * (n - 2) * (n - 3))

        Ymean = p2 * sumB
        wedgesum = (sumBrow2 - 2 * sumB2)
        Ymeansq = p2 * sumB2 + p3 * wedgesum + p4 * (sumB**2 - sumB2 - wedgesum)
        Ystd = np.sqrt(Ymeansq - Ymean**2)

        adjsum = np.dot(x, np.dot(A, x)) / 2 - np.dot(x, deg)**2 / (4 * m) + np.dot(x, deg**2) / (4 * m)
        beta = 0.9
        delta = 0.8
        score = k**(-delta) * (adjsum - (Ymean + 1.28 * Ystd) * (k / s)**beta)

    return score
