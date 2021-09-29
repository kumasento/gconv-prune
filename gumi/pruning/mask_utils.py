""" Calculators for pruning masks from pre-trained data.

Author: Ruizhe Zhao
Date: 13/02/2019
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

__all__ = ["create_mbm_mask"]


def get_criterion(W, crit_type=None):
    """ Compute the magnitude-based criterion on W.

        Returns:
            A 2d torch.Tensor of (F, C)
    """
    assert isinstance(W, torch.Tensor)
    assert W.dim() == 4

    if crit_type == "taylor":
        return get_taylor_criterion(W)

    kernel_dims = (2, 3)
    C = torch.norm(W, dim=kernel_dims)

    return C


def get_taylor_criterion(W):
    """ Get the pruning criterion based on taylor expansion.
        It requires that W.grad is accessible.
    """
    assert isinstance(W, torch.Tensor)
    assert W.grad is not None
    assert W.dim() == 4

    # first-order term of the taylor expansion
    C = torch.mul(W.grad, W)

    # L1-norm taken on the taylor criterion
    C = torch.abs(C).sum(dim=(2, 3))

    return C


def _get_numpy(tensor):
    """ Get the numpy.ndarray from a (possible) ND array. """
    if isinstance(tensor, torch.Tensor):
        X = tensor.clone()  # TODO: is this necessary?
        return X.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor

    raise TypeError("Cannot handle tensor of type: {}".format(type(tensor)))


def get_sort_perm(C, dim=0):
    """ Get the permutation indices that can sort the given
    criterion. Sorting is done by comparing the sum of all
    other dimensions expect dim.

  Returns:
    A numpy.ndarray of permuted indices.
  """
    assert dim < len(C.shape)

    C = _get_numpy(C)
    # dimensions beside dim
    sum_dims = tuple([d for d in range(len(C.shape)) if d != dim])

    return np.argsort(C.sum(axis=sum_dims))


def group_sort(C, G, num_iters=1, min_g=0):
    """ Sort the criterion matrix by the last group of
    rows and columns.

    The whole recursion works like this:
    --> sort by the last group of columns and rows, collect
        their permutation.
    --> pass the sorted C[:C.shape[0]-g_out, :C.shape[1]-g_in]
        (= C') into the next step.
    --> the returned result will be a sorted C', the
        permutation of this submatrix's indices
    --> update the matrix and indices to be returned

  Args:
  """
    assert isinstance(C, np.ndarray)

    c_out, c_in = C.shape
    g_out, g_in = c_out // G, c_in // G

    gnd_in, gnd_out = np.arange(c_in), np.arange(c_out)

    for g in reversed(range(G)):
        if g < min_g:
            break

        # heuristic method
        for _ in range(num_iters):

            # first sort the columns by the sum of the last row group
            r_lo, r_hi = g_out * g, g_out * (g + 1)
            c_lo, c_hi = g_in * g, g_in * (g + 1)

            # get the current sorting result
            # C will be updated every time
            C_ = C[gnd_out, :][:, gnd_in]

            # crop the matrix
            C_ = C_[:r_hi, :c_hi]
            # print(C_)

            # rows and cols for sorting
            rows = C_[r_lo:, :]
            perm_cols = np.argsort(rows.sum(axis=0))

            cols = C_[:, perm_cols][:, c_lo:]
            perm_rows = np.argsort(cols.sum(axis=1))

            # print(rows, rows.sum(axis=0))
            # print(cols, cols.sum(axis=1))
            # print(perm_rows, perm_cols)

            # update gnd_in and gnd_out
            gnd_in[:c_hi] = gnd_in[:c_hi][perm_cols]
            gnd_out[:r_hi] = gnd_out[:r_hi][perm_rows]

    return gnd_in, gnd_out


def permute_criterion(C, method=None, G=None, num_iters=1):
    """ Permute criterion by a specific method.

    Available methods:
        'SAME': don't change the order
        'SORT': sort the criterion by sum-of-others 
    Returns:
        A permuted criterion (np.ndarray), indices of C_in, indices of C_out 
    """
    C = _get_numpy(C)
    if method is None or method == "SAME":
        # NOTE: C_in first
        return C, np.arange(C.shape[1]), np.arange(C.shape[0])
    elif method == "SORT":
        ind_in, ind_out = get_sort_perm(C, dim=1), get_sort_perm(C, dim=0)
        return (C[ind_out, :])[:, ind_in], ind_in, ind_out
    elif method == "GRPS":
        assert isinstance(G, int) and G > 0
        ind_in, ind_out = group_sort(C, G, num_iters=num_iters)
        return C[ind_out, :][:, ind_in], ind_in, ind_out

    raise ValueError('Cannot recognize method: "{}"'.format(method))


def run_mbm_core(W, G, perm="GRPS", num_iters=10, crit_type=None):
    """ Core MBM algorithm. """

    # Get the criterion matrix C. If the input matrix is 2-D, just use
    # that as C; otherwise, calculate C by `get_criterion`.
    if len(W.shape) == 2:
        C = W
    else:
        C = get_criterion(W, crit_type=crit_type)

    # Normalise C.
    C /= torch.norm(C)

    orig_crit = C.clone().detach().cpu()
    # permute the criterion by given permutation method
    crit, ind_in, ind_out = permute_criterion(C, method=perm, G=G, num_iters=num_iters)

    # crit has been permuted
    c_out, c_in = crit.shape
    g_out, g_in = c_out // G, c_in // G  # group shape

    # compute the cost matrix for MBM
    cost = crit.reshape([G, g_out, G, g_in])
    cost = cost.transpose([0, 2, 1, 3])  # transpose g_in, G
    cost = cost.sum(axis=(2, 3))  # final cost matrix

    # run MBM
    row_ind, col_ind = linear_sum_assignment(-cost)

    # recover to the indices in the original criterion matrix
    gnd_in = []
    gnd_out = []

    # Figure out the indices for each group
    for r, c in zip(row_ind, col_ind):
        rs, cs = r * g_out, c * g_in
        # TODO: necessary to sort? just look better
        gnd_in.append(ind_in[cs : (cs + g_in)])
        gnd_out.append(ind_out[rs : (rs + g_out)])

    # sort by the min index of each group in gnd_in
    gnd_in = np.array(gnd_in, dtype=np.int32)
    gnd_out = np.array(gnd_out, dtype=np.int32)
    # indices of sorted groups
    grps = np.argsort(np.min(gnd_in, axis=1))

    cost_ = cost[row_ind, col_ind].sum()

    return gnd_in[grps, :], gnd_out[grps, :], cost.sum() - cost_, orig_crit


def run_mbm(W, G, perm="GRPS", num_iters=10, crit_type=None, normalized=False):
    """ Core MBM algorithm. """
    gnd_in, gnd_out, cost, _ = run_mbm_core(
        W, G, perm=perm, num_iters=num_iters, crit_type=crit_type
    )

    return gnd_in, gnd_out, cost


def create_mbm_mask(W, G, **kwargs):
    """ Create a maximal bipartite matched mask for given tensor.
    Args:
        W(torch.Tensor): should be the weight
        G(int): number of groups
    Returns:
        Another torch.Tensor for the mask
    """
    # run MBM
    gnd_in, gnd_out, _ = run_mbm(W, G, **kwargs)

    # create the mask
    mask = torch.zeros(W.shape[:2])  # same shape as the criterion
    for x, y in zip(gnd_out, gnd_in):
        ind = tuple([np.tile(x, len(y)), np.repeat(y, len(x))])
        mask[ind] = 1.0

    return mask


def create_pruning_mask(W, G, perm="GRPS", num_iters=10):
    """ Create the pruning mask.
        This API is explicitly used by gopt_v3,
        may become mainstream in the future.
    """
    gnd_in, gnd_out, cost, crit = run_mbm_core(W, G, perm=perm, num_iters=num_iters)

    # create the mask
    mask = torch.zeros(W.shape[:2])  # same shape as the criterion
    for x, y in zip(gnd_out, gnd_in):
        ind = tuple([np.tile(x, len(y)), np.repeat(y, len(x))])
        mask[ind] = 1.0

    return mask, crit, cost
