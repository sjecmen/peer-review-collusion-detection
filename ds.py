import gurobipy as gp
import numpy as np

def densest_subgraph(G):
    assert all([G[i, i] == 0 for i in range(G.shape[0])])

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    X = m.addMVar(G.shape, lb=0, ub=G, obj=1)
    Y = m.addMVar(G.shape[0], lb=0, obj=0)
    m.modelSense = gp.GRB.MAXIMIZE
    m.addConstr(Y @ np.ones(G.shape[0]) <= 1)
    m.addConstrs(X[i, :] <= Y[i] for i in range(G.shape[0]))
    m.addConstrs(X[:, j] <= Y[j] for j in range(G.shape[0]))

    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    Y_ = Y.x

    r_values =  np.unique(Y_)
    f_values = []
    for r in r_values:
        S = np.argwhere(Y_ >= r).flatten()
        f = G[np.ix_(S, S)].sum() / S.size
        f_values.append((f, S))
    f_star, S_star = max(f_values)
    assert np.isclose(f_star, m.ObjVal), f'{f_star}, {m.ObjVal}'

    return S_star, ''


