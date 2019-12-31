import torch
import numpy as np
from scipy import sparse
import bottleneck as bn
'''
Adapted from : https://github.com/samlobel/RaCT_CF/blob/master/utils/evaluation_functions.py
'''

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100, input_batch=None, normalize=True):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    If normalize is set to False, then we actually return DCG, not NDCG.
    '''
    if input_batch is not None:
        X_pred[input_batch.nonzero().chunk(chunks=2,dim=1)] = -np.inf
   
    batch_users = X_pred.shape[0]
    # Get the indexes of the top K predictions.
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    # Get only the top k predictions.
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    # Get sorted index...
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    # Get sorted index...
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = torch.tensor(1. / np.log2(np.arange(2, k + 2)),dtype=torch.float32)
    # You add up the ones you've seen, scaled by their discount...
    # top_k_results = heldout_batch[np.arange()]
    maybe_sparse_top_results = heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk]
    try:
        top_results = maybe_sparse_top_results.toarray()
    except:
        top_results = maybe_sparse_top_results

    try:
        number_non_zero = heldout_batch.getnnz(axis=1)
    except:
        number_non_zero = ((heldout_batch > 0) * 1).sum(axis=1)

    DCG = (top_results * tp).sum(axis=1)
    # DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
    #                      idx_topk].toarray() * tp).sum(axis=1)
    # Gets denominator, could be the whole sum, could be only part of it if there's not many.
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in number_non_zero])

    IDCG = np.maximum(0.1, IDCG) #Necessary, because sometimes you're not given ANY heldout things to work with. Crazy...
    IDCG = torch.tensor(IDCG,dtype=torch.float32)
    # IDCG = np.array([(tp[:min(n, k)]).sum()
    #                  for n in heldout_batch.getnnz(axis=1)])
    # to_return = DCG / IDCG
    # if np.any(np.isnan(to_return)):
    #     print("bad?!")
    #     import ipdb; ipdb.set_trace()
    #     print("dab!?")
    if normalize:
        result = (DCG / IDCG)
    else:
        result = DCG
#     result = result.astype(np.float32)
    return result
