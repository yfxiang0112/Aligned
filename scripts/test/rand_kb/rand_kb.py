from scipy.sparse import load_npz, coo_matrix, save_npz
import numpy as np

data_name = 'norman'
KB_P = load_npz(f'rules/human/{data_name}_KB_P.npz').toarray()
KB_N = load_npz(f'rules/human/{data_name}_KB_N.npz').toarray()
p_pos = np.count_nonzero(KB_P) / (KB_P.shape[0]*KB_P.shape[1])
p_neg = np.count_nonzero(KB_N) / (KB_N.shape[0]*KB_N.shape[1])

KB_P_rand = np.random.choice([1,0], size=KB_P.shape, p=[p_pos,1-p_pos])
KB_N_rand = np.random.choice([1,0], size=KB_N.shape, p=[p_neg,1-p_neg])

save_npz(f'scripts/test/rand_kb/{data_name}_KB_N_rand.npz',coo_matrix(KB_P_rand))
save_npz(f'scripts/test/rand_kb/{data_name}_KB_N_rand.npz',coo_matrix(KB_P_rand))
