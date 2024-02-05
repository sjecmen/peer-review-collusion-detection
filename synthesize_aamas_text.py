import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import utils

'''
Construct and save synthetic similarities for the AAMAS dataset. Requires that you download the
ICLR text-similarities provided by (Xu et al., 2018): 
    https://github.com/xycforgithub/StrategyProof_Conference_Review
'''

easy_mean = 0.80
hard_mean = 0.62
easy_ci = [0.72, 0.87]
hard_ci = [0.54, 0.69]

B = utils.load_data_aamas()['bid_matrix']
T_iclr = np.load('iclr2018.npz')['similarity_matrix'].flatten()
mu0, std = norm.fit(T_iclr)
print(mu0, std)

delta12 = norm.ppf(hard_mean, scale=(np.sqrt(2) * std))

counts = [np.sum(B == x) for x in [0, 1, 2]]
prop1 = counts[1] / (counts[1] + counts[2])
prop2 = counts[2] / (counts[1] + counts[2])

def f(mu1):
    p01 = norm.cdf(mu1-mu0, scale=(np.sqrt(2) * std))
    p02 = norm.cdf(delta12+mu1-mu0, scale=(np.sqrt(2) * std))
    lhs = (p01 * prop1) + (p02 * prop2)
    rhs = easy_mean
    return lhs - rhs
sol = root_scalar(f, bracket=(0, 1))
assert sol.converged
mu1 = sol.root
assert abs(f(mu1)) < 1e-5
mu2 = mu1 + delta12

p12 = norm.cdf(0, loc=(mu1-mu2), scale=(np.sqrt(2) * std))
p01 = norm.cdf(0, loc=(mu0-mu1), scale=(np.sqrt(2) * std))
p02 = norm.cdf(0, loc=(mu0-mu2), scale=(np.sqrt(2) * std))
p0x = (p01 * prop1) + (p02 * prop2)
assert abs(p12 - hard_mean) < 1e-5
assert abs(p0x - easy_mean) < 1e-5

rng = np.random.default_rng(0)
mus = [mu0, mu1, mu2]
print(mus)
samples = [norm.rvs(loc=mus[x], scale=std, size=counts[x], random_state=rng) for x in [0, 1, 2]]

T = np.zeros(B.shape, dtype=float)
for x in [0, 1, 2]:
    T[B == x] = samples[x]
T = np.clip(T, 0, 1)

Bt_easy = np.zeros((B.shape[1], B.shape[0], B.shape[0]), dtype=bool)
Bt_hard = np.zeros((B.shape[1], B.shape[0], B.shape[0]), dtype=bool)
for r in range(B.shape[1]):
    Bv = B[:, r]
    Bt_easy[r, :, :] = np.outer((Bv > 0), (Bv == 0))
    Bt_hard[r, :, :] = np.outer((Bv == 2), (Bv == 1))

easy_correct = 0
hard_correct = 0
for r in range(T.shape[1]):
    Tv = T[:, r]
    Tm = (np.subtract.outer(Tv, Tv) >= 0)
    easy_correct += (Bt_easy[r] & Tm).sum()
    hard_correct += (Bt_hard[r] & Tm).sum()
easy_f = easy_correct / Bt_easy.sum()
hard_f = hard_correct / Bt_hard.sum()

print(f'easy={easy_f:.02f}, hard={hard_f:.02f}')
mu, std = norm.fit(T)
print(mu, std)
np.save('datasets/aamas_text.npy', T)
