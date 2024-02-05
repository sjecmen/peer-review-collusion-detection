import json
import numpy as np
import random
import utils

'''
Compute and save authorship matrix for aamas_sub3 and wu datasets. This file requires
that you download the original dataset provided by (Wu et al., 2021):
    https://github.com/facebookresearch/secure-paper-bidding
'''

# Construct and save the authorship for wu 
input_dir = 'secure-paper-bidding-main/data/raw_data/'
with open(input_dir + 'papers_dictionary.json') as f:
    pap_dict = json.load(f)
with open(input_dir + 'reviewers_dictionary.json') as f:
    rev_dict = json.load(f)
A = np.zeros((len(pap_dict), len(rev_dict)))

author_to_papers = {}
for pid, pap in pap_dict.items():
    for author in pap['authors']:
        if author in author_to_papers:
            author_to_papers[author].append(pid)
        else:
            author_to_papers[author] = [pid]

for rid, rev in rev_dict.items():
    if rev['name'] in author_to_papers:
        for pid in author_to_papers[rev['name']]:
            A[int(pid), int(rid)] = 1
np.save('datasets/wu_authorship.npy', A)


# Construct and save the authorship for aamas_sub3 
data = utils.load_data_aamas()
A = data['author_matrix']
rng = np.random.default_rng(seed=0)
A_sub = np.zeros_like(A)
for p in range(data['n_pap']):
    s = A[p, :].sum()
    if s <= 3:
        A_sub[p, :] = A[p, :]
    else:
        author_sample = rng.choice(A.shape[1], size=3, replace=False, p=(A[p, :] / s))
        A_sub[p, author_sample] = 1
assert (A_sub.sum(axis=1) <= 3).all()
assert (A_sub <= A).all()
np.save('datasets/aamas_authorship.npy', A_sub)


