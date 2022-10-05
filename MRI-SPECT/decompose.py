import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg
import utils
def edge_preserving_decompose(img, iter=4, k=3, k_step=8):
    M = img.astype('float32')
    Ds = []
    for i in range(iter):
        #print("decompose iteration %s" % (i + 1))
        min_mask, max_mask = local_extrema(M, k)
        envelope_bottom = envelope_bound_interpolation(M, min_mask)
        envelope_top = envelope_bound_interpolation(M, max_mask)
        new_M = (envelope_top + envelope_bottom) / 2
        del envelope_bottom, envelope_top
        
        D = M - new_M
        Ds.append(D)

        M = new_M
        k += k_step

    return clip_and_convert_to_uint8(M), Ds


def edge_preserving_decompose1(img, iter=4, k=3, k_step=8):
    M = img.astype('float32')
    Ds = []
    Ms=[]
    for i in range(iter):
        print("decompose iteration %s" % (i + 1))
        min_mask, max_mask = local_extrema(M, k)
        envelope_bottom = envelope_bound_interpolation(M, min_mask)
        envelope_top = envelope_bound_interpolation(M, max_mask)
        new_M = (envelope_top + envelope_bottom) / 2
        del envelope_bottom, envelope_top

        D = M - new_M
        Ds.append(D)
        Ms.append(new_M)
        M = new_M
        k += k_step

    return Ms, Ds

def clip_and_convert_to_uint8(img):
    return np.clip(img, 0, 255).astype('uint8')


def envelope_bound_interpolation(M, extrema_mask):

    neighbor_k = 11
    padded_M = pad_reflect(M, neighbor_k)
    padded_extrema_mask, padding = pad_ones(extrema_mask, neighbor_k)
    padded_constraint_idx = np.nonzero(padded_extrema_mask.ravel())[0]

    #since the implementation difficulty, A must introduces redundant padded variables
    A = compute_A(M, padded_M, padded_extrema_mask, neighbor_k)

    b = np.zeros(A.shape[0], dtype='float32')
    b[padded_constraint_idx] = padded_M.ravel()[padded_constraint_idx]
    #print('start solving sparse linear system')
    E = linalg.lsmr(A, b)[0]
    del A, b
    return E.reshape(padded_M.shape)[padding: -padding, padding: -padding]


EPSILON = 1e-6
def compute_A(M, padded_M, padded_extrema_mask, neighbor_k):
    center_idx = (neighbor_k * neighbor_k) // 2
    padded_pixel_num = padded_M.size
    conv_idx_vol = conv_idx_volume(padded_M, M.shape, neighbor_k)

    neighbor_vol = np.take(padded_M, conv_idx_vol)
    variance = np.var(neighbor_vol, 0)  # note the variance includes the padded pixels
    variance[variance < EPSILON] = EPSILON

    conv_idx_vol = pad_vol_zeros(conv_idx_vol, neighbor_k)
    conv_idx_vol[center_idx] = np.arange(padded_pixel_num, dtype='int').reshape(conv_idx_vol.shape[1:])
    A_col_idx = conv_vol_ravel(conv_idx_vol)
    del conv_idx_vol

    A_row_idx = np.tile(np.arange(padded_pixel_num)[:, np.newaxis], [1, neighbor_k * neighbor_k]).ravel() # conv_idx_vol.shape[0] == k**2

    fill_A_vol = np.exp(-((M - neighbor_vol) ** 2) / (2 * variance)) # compute w(r, s), and times -1
    del variance
    fill_A_vol = -fill_A_vol / np.sum(fill_A_vol, axis=0)
    fill_A_vol = pad_vol_zeros(fill_A_vol, neighbor_k)
    fill_A_vol[:, padded_extrema_mask] = 0 # constraints do not need to be estimated
    fill_A_vol[center_idx] = 1 #set the center pixel's coefficient to 1

    #note, A contains pixels from padding. However, they're constrained
    #since removeing these variable is hard in the sparse matrix form
    A = sparse.bsr_matrix((conv_vol_ravel(fill_A_vol), (A_row_idx, A_col_idx)), shape=(padded_pixel_num, padded_pixel_num))
    return A

def conv_vol_ravel(vol):
    return vol.swapaxes(1, 2).swapaxes(2, 0).ravel()

def local_extrema(img, k):
    vol = conv_volume(img, k)
    conv_var = np.var(vol, axis=0)
    # utils.plot_hist(conv_var.ravel(), [0, np.max(conv_var)])
    flat_area_mask = conv_var <= 25
    center_idx = (k * k) // 2

    lt_cent_statistics = np.sum(vol < vol[center_idx], axis=0)
    gt_cent_statistics = np.sum(vol > vol[center_idx], axis=0)

    EXTREMA_CRITERIA = k ** 2 / 3
    # EXTREMA_CRITERIA = k-1 # in original paper is k - 1, it will produce too much unknown

    # return gt_cent_statistics <= EXTREMA_CRITERIA, lt_cent_statistics <= EXTREMA_CRITERIA
    return (np.logical_or(gt_cent_statistics <= EXTREMA_CRITERIA, flat_area_mask),
            np.logical_or(lt_cent_statistics <= EXTREMA_CRITERIA, flat_area_mask))

def conv_volume(img, k):
    padded_img = pad_reflect(img, k)
    volume = np.take(padded_img, conv_idx_volume(padded_img, img.shape, k))
    return volume

def pad_ones(img, k):
    padding = k // 2
    padded_img = np.pad(img, padding, 'constant', constant_values=1)
    return padded_img, padding

def pad_reflect(img, k):
    padding = k // 2
    padded_img = np.pad(img, padding, 'reflect')

    return padded_img

def pad_vol_zeros(vol, k):
    padding = k // 2
    return np.pad(vol, [[0, 0], [padding, padding], [padding, padding]], 'constant', constant_values=0)

def conv_idx_volume(padded_img, img_shape, k):
    H, W = padded_img.shape

    # start_idx_slice = np.arange(H * W).reshape((H, W))[:-k + 1, :-k + 1]
    # use broadcast trick: column vector + row vector = matrix
    start_idx_slice = np.arange(img_shape[0], dtype='int')[:, np.newaxis] * W + np.arange(img_shape[1], dtype='int')
    conv_idx = np.arange(k, dtype='int')[:, np.newaxis] * W + np.arange(k, dtype='int')
    conv_idx_3d = conv_idx.ravel()[:, np.newaxis, np.newaxis]
    # use broadcast trick: 3d column vector + matrix = volume
    return conv_idx_3d + start_idx_slice
def adjust_constrast(original_M, new_M):
    STATISTICS_RANGE = 10
    o_max = np.max(original_M)
    o_min = np.min(original_M)
    n_max = np.max(new_M)
    n_min = np.min(new_M)
    o_max_mean = original_M[original_M >= (o_max - STATISTICS_RANGE)].mean()
    o_min_mean = original_M[original_M <= (o_min + STATISTICS_RANGE)].mean()
    n_max_mean = new_M[new_M >= (n_max - STATISTICS_RANGE)].mean()
    n_min_mean = new_M[new_M <= (n_min + STATISTICS_RANGE)].mean()
    o_range = o_max_mean - o_min_mean
    n_range = n_max_mean - n_min_mean
    new_M = new_M * (o_range / n_range)
    return clip_and_convert_to_uint8((o_min - np.min(new_M)) + new_M)



