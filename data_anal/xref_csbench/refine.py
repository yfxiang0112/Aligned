import torch
import numpy as np
from egoal.reasoner import RegulatoryKB
from egoal.learner_refl import ReflectLearner
from scipy.sparse import base, load_npz, coo_matrix, save_npz
import json
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, confusion_matrix, precision_score
import glob
import os

data_name = 'norman'

xref_string = load_npz('data_anal/xref_csbench/networks/string.npz').toarray()
xref_corum = load_npz('data_anal/xref_csbench/networks/corum.npz').toarray()
xref_lr = load_npz('data_anal/xref_csbench/networks/lr_pairs.npz').toarray()
xref_chipseq = load_npz('data_anal/xref_csbench/networks/chipseq.npz').toarray()

directory = f'data_anal/refine/models/'
file_pattern = '*.npz'
model_files = glob.glob(os.path.join(directory, file_pattern))
model_files.sort()  # Sort for consistent ordering

KB_orig = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                  device='cuda')
KB_orig.closure_(T=5, closure_type='naive')
KB_P_indirect = torch.clip(KB_orig.KB_P - KB_orig.Regu_P_0,0,1).to('cpu')
KB_N_indirect = torch.clip(KB_orig.KB_N - KB_orig.Regu_N_0,0,1).to('cpu')
KB_P_direct = KB_orig.Regu_P_0.to('cpu')
KB_N_direct = KB_orig.Regu_N_0.to('cpu')

KB = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                  device='cpu')

KB_baseline = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                  device='cpu')


#seed = 42
#incomp_p_pth = 'scripts/klg_refine/kb/incomp_P.npz'
#incomp_n_pth = 'scripts/klg_refine/kb/incomp_N.npz'

#for file in model_files:
for p_incompl in [0., .05, .1, .2, .3, .4, .5]:
    #KB.load(file)
    KB.load(f'data_anal/refine/models/restored_{p_incompl}_mix_1.npz')
    KB_baseline.load(f'data_anal/refine/models/restored_{p_incompl}_mix_baseline_1.npz')

    shortcut = torch.count_nonzero(((KB.Regu_P_0!=0)&(KB_P_indirect!=0)))
    direct = torch.count_nonzero(((KB.Regu_P_0!=0)&(KB_P_direct!=0)))
    total = torch.count_nonzero(KB.Regu_P_0)
    #print(f'p={p_incompl}, learned shortcuts: {float(shortcut/total):.3f}, direct: {float(direct/total):.3f}')

    baseline_shortcut = torch.count_nonzero(((KB_baseline.Regu_P_0!=0)&(KB_P_indirect!=0)))
    baseline_direct = torch.count_nonzero(((KB_baseline.Regu_P_0!=0)&(KB_P_direct!=0)))
    baseline_total = torch.count_nonzero(KB_baseline.Regu_P_0)
    #print(f'p={p_incompl}, baseline shortcuts: {float(baseline_shortcut/baseline_total):.3f}, direct: {float(baseline_direct/baseline_total):.3f}\n')
    print(f'{p_incompl} | {100*direct/total: .1f}% | {100*shortcut/total: .1f}% | {100*baseline_direct/baseline_total: .1f}% | {100*baseline_shortcut/baseline_total: .1f}%')


#for p_incompl in [0., .05, .1, .2, .3, .4, .5]:
for p_incompl in [0., .05, .1, .3, .5]:
    KB.load(f'data_anal/refine/models/restored_{p_incompl}_mix_1.npz')
    KB_baseline.load(f'data_anal/refine/models/restored_{p_incompl}_mix_baseline_1.npz')

    #test = []
    #baseline = []
    for ref, name in [(xref_string, 'string'),
                      (xref_corum,  'corum'),
                      (xref_lr,     'lr'),
                      (xref_chipseq,'chipseq')]:
        #print(f'\np={p_incompl} on {name} database:')

        pred = torch.clamp(torch.abs(KB.Regu_P_0)+torch.abs(KB.Regu_N_0), 0,1).flatten()
        w = np.count_nonzero(ref) / (ref.shape[0]*ref.shape[1])

        pred_baseline = torch.clamp(torch.abs(KB_baseline.Regu_P_0)+torch.abs(KB_baseline.Regu_N_0), 0,1).flatten()
        #w_baseline = torch.count_nonzero(pred_baseline).item() / len(pred_baseline)

        f1_aligned = precision_score(ref.flatten(), pred, sample_weight=np.where(pred==0, w, 1-w))
        f1_baseline = precision_score(ref.flatten(), pred_baseline, sample_weight=np.where(pred_baseline==0, w, 1-w))

        #test.append(f1_aligned)
        #baseline.append(f1_baseline)
        #print(p_incompl, 'ALIGNED ', [f'{x:.4f}' for x in test])
        #print(p_incompl, 'baseline', [f'{x:.4f}' for x in baseline])
        print(f'ALIGNED: {f1_aligned: .3f}, baseline: {f1_baseline: .3f}')
        print(confusion_matrix(ref.flatten(), pred))


        #' mask & save incomplete KB '
        #save_npz(incomp_p_pth, coo_matrix(np.where(remove_mask, 0, 
        #    np.where(positive_mask, 1, R_P))))
        #save_npz(incomp_n_pth, coo_matrix(np.where(remove_mask, 0, 
        #    np.where(negative_mask, 1, R_N))))

        #reasoner_incomp = RegulatoryKB(
        #        pos_trn_pth= incomp_p_pth,
        #        neg_trn_pth= incomp_n_pth,
        #        device='cpu')


        print('\n----------')
