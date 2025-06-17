from typing import Tuple
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ot
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import lsqr
from tqdm import tqdm
import time
import jax.numpy as jnp
import torch
from treeOT import *
import pickle as pkl
from scipy.sparse.linalg import svds
from jax import vmap, random

def regularization_matrix(
    A: torch.Tensor,
    p: int,
    dtype: str,
    device: str,
) -> torch.Tensor:
    """From Huizing et al., 2022 (https://github.com/CSDUlm/wsingular/tree/main)
    Return the regularization matrix

    Args:
        A (torch.Tensor): The dataset, with samples as rows
        p (int): order of the norm
        dtype (str): The dtype to be returned
        device (str): The device to be returned

    Returns:
        torch.Tensor: The regularization matrix
    """

    # Return the pairwise distances using torch's `cdist`.
    return torch.cdist(A, A, p=p).to(dtype=dtype, device=device)


def hilbert_distance(D_1: torch.Tensor, D_2: torch.Tensor) -> float:
    """From Huizing et al., 2022 (https://github.com/CSDUlm/wsingular/tree/main)
    Compute the Hilbert distance between two distance-like matrices.

    Args:
        D_1 (torch.Tensor): The first matrix
        D_2 (torch.Tensor): The second matrix

    Returns:
        float: The distance
    """

    # Perform some sanity checks.
    assert torch.sum(D_1 < 0) == 0  # positivity
    assert torch.sum(D_2 < 0) == 0  # positivity
    assert D_1.shape == D_2.shape  # same shape

    try:
        # Get a mask of all indices except the diagonal.
        idx = torch.eye(D_1.shape[0]) != 1
    
        # Compute the log of D1/D2 (except on the diagonal)
        div = torch.log(D_1[idx] / D_2[idx])
    except IndexError: # if we have a vector rather than a matrix
        div = torch.log(D_1 / D_2)
    # Return the Hilbert projective metric.
    return float((div.max() - div.min()).cpu())


def hilbert_distance_jax(D_1: jnp.ndarray, D_2: jnp.ndarray) -> float:
    """Compute the Hilbert distance between two distance-like matrices that are Jax ndarrays.

    Args:
        D_1 (torch.Tensor): The first matrix
        D_2 (torch.Tensor): The second matrix

    Returns:
        float: The distance
    """

    # Perform some sanity checks.
    assert jnp.sum(D_1 < 0) == 0  # positivity
    assert jnp.sum(D_2 < 0) == 0  # positivity
    assert D_1.shape == D_2.shape  # same shape

    div = jnp.log(D_1 / D_2)
    # Return the Hilbert projective metric.
    return (div.max() - div.min())


def normalize_dataset(
    dataset: torch.Tensor,
    dtype: str,
    device: str,
    normalization_steps: int = 1,
    small_value: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """From Huizing et al., 2022 (https://github.com/CSDUlm/wsingular/tree/main)
    Normalize the dataset and return the normalized dataset A and the transposed dataset B.

    Args:
        dataset (torch.Tensor): The input dataset, samples as rows.
        normalization_steps (int, optional): The number of Sinkhorn normalization steps. For large numbers, we get bistochastic matrices. Defaults to 1 and should be larger or equal to 1.
        small_value (float): Small addition to the dataset to avoid numerical errors while computing OT distances. Defaults to 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The normalized matrices A and B.
    """

    # Perform some sanity checks.
    assert len(dataset.shape) == 2  # correct shape
    assert torch.sum(dataset < 0) == 0  # positivity
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    # Do a first normalization pass for A
    A = dataset / dataset.sum(1).reshape(-1, 1)
    A += small_value
    A /= A.sum(1).reshape(-1, 1)

    # Do a first normalization pass for B
    B = dataset.T / dataset.T.sum(1).reshape(-1, 1)
    B += small_value
    B /= B.sum(1).reshape(-1, 1)

    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        A, B = B.T / B.T.sum(1).reshape(-1, 1), A.T / A.T.sum(1).reshape(-1, 1)

    return A.to(dtype=dtype, device=device), B.to(dtype=dtype, device=device)


def normalize_dataset_jax(
    dataset: jnp.array,
    dtype: str,
    device: str,
    normalization_steps: int = 1,
    small_value: float = 1e-6):
    """Normalize the dataset and return the normalized dataset A and the transposed dataset B.

    Args:
        dataset (torch.Tensor): The input dataset, samples as rows.
        normalization_steps (int, optional): The number of Sinkhorn normalization steps. For large numbers, we get bistochastic matrices. Defaults to 1 and should be larger or equal to 1.
        small_value (float): Small addition to the dataset to avoid numerical errors while computing OT distances. Defaults to 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The normalized matrices A and B.
    """

    # Perform some sanity checks.
    assert len(dataset.shape) == 2  # correct shape
    assert jnp.sum(dataset < 0) == 0  # positivity
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    # Do a first normalization pass for A
    A = dataset / dataset.sum(1).reshape(-1, 1)
    A += small_value
    A /= A.sum(1).reshape(-1, 1)

    # Do a first normalization pass for B
    B = dataset.T / dataset.T.sum(1).reshape(-1, 1)
    B += small_value
    B /= B.sum(1).reshape(-1, 1)

    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        A, B = B.T / B.T.sum(1).reshape(-1, 1), A.T / A.T.sum(1).reshape(-1, 1)

    return A, B


def generate_torus_data(n_samples = 1000, n_features = 1000, perm=True, dtype=jnp.float32):
    """ 
    Generate unimodal torus data (adapted from Huizing et al., 2022)
    Optionally permute
    """
    samp_arr = jnp.linspace(0,n_samples,n_samples,endpoint=False)
    feat_arr = jnp.linspace(0,n_features,n_features,endpoint=False)

    dataset = jnp.abs(samp_arr[:,None]/n_samples - feat_arr[None,:]/n_features % 1)

    # iterate over the features and samples.
    dataset = np.zeros((n_samples, n_features), dtype=dtype)
    for i in range(n_samples):
        for j in range(n_features):

            # fill the dataset with translated histograms.
            dataset[i, j] = i/n_samples - j/n_features
            dataset[i, j] = abs(dataset[i, j] % 1)

    # take the distance to 0 on the torus
    dataset = jnp.fmin(dataset, 1 - dataset)

    # make it a gaussian
    dataset = jnp.exp(-(dataset)**2 / 0.1)

    # define the dimensions of the problem
    n_samples, n_features = dataset.shape
    return dataset


def display_cost(C,D,n_samples,n_features,name=None):
    """From Huizing et al., 2022 (https://github.com/CSDUlm/wsingular/tree/main)"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Wasserstein Singular Vectors')

    axes[0].set_title('Distance between samples.')
    axes[0].imshow(D)
    axes[0].set_xticks(range(0, n_samples, 5))
    axes[0].set_yticks(range(0, n_samples, 5))

    axes[1].set_title('Distance between features.')
    axes[1].imshow(C)
    axes[1].set_xticks(range(0, n_features, 5))
    axes[1].set_yticks(range(0, n_features, 5))
    if name is not None:
        plt.savefig('images/cost_'+name+'.png',dpi=150)
    else:
        plt.show()
    return

def silhouette(D: jnp.array, labels) -> float:
    """Return the average silhouette score, given a distance matrix and labels.
    From Huizing et al., 2022 (https://github.com/CSDUlm/wsingular/tree/main)

    Args:
        D (torch.Tensor): Distance matrix n*n
        labels (Iterable): n labels

    Returns:
        float: The average silhouette score
    """

    # Perform some sanity checks.
    assert len(D.shape) == 2  # correct shape
    assert jnp.sum(D < 0) == 0  # positivity

    return silhouette_score(D, labels, metric="precomputed")


def timer(prev_time,comment=''):
    print(comment,'gap: ',str(time.time()-prev_time))
    return time.time()


def extract_basis(matrix):
    """
    Extract a basis for the column space of the given matrix.
    
    Parameters:
    matrix (numpy.ndarray): The input matrix.
    
    Returns:
    numpy.ndarray: A matrix whose columns form a basis for the column space of the input matrix.
    """
    # Perform Gaussian elimination to get the REF
    ref_matrix = np.array(matrix, dtype=np.float64)
    (rows, cols) = ref_matrix.shape
    
    pivot_columns = []
    row = 0
    
    for col in range(cols):
        # Find the pivot row for this column
        pivot_row = None
        for r in range(row, rows):
            if ref_matrix[r, col] != 0:
                pivot_row = r
                break
                
        if pivot_row is None:
            continue  # No pivot in this column
        
        # Swap the current row with the pivot row
        ref_matrix[[row, pivot_row]] = ref_matrix[[pivot_row, row]]
        
        # Normalize the pivot row
        ref_matrix[row] = ref_matrix[row] / ref_matrix[row, col]
        
        # Eliminate the entries below the pivot
        for r in range(row + 1, rows):
            ref_matrix[r] = ref_matrix[r] - ref_matrix[r, col] * ref_matrix[row]
        
        pivot_columns.append(col)
        row += 1
    
    # Extract the pivot columns from the original matrix
    basis = matrix[:, pivot_columns]
    
    return basis, pivot_columns


def extract_basis_sparse_lsqr(matrix):
    """
    Extract a basis for the column space of the given sparse matrix using LSQR.
    
    Parameters:
    matrix (scipy.sparse.csr_matrix): The input sparse matrix.
    
    Returns:
    numpy.ndarray: A matrix whose columns form a basis for the column space of the input matrix.
    numpy.ndarray: Indices of the columns that form the basis.
    """
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    
    m, n = matrix.shape
    basis_vectors = []
    basis_indices = []
    
    for i in tqdm(range(m)):
        e = np.zeros(m)
        e[i] = 1
        x = lsqr(matrix, e)[0]
        if np.linalg.norm(x) > 1e-10:
            basis_vectors.append(x)
            basis_indices.append(i)
    
    basis = matrix[:, basis_indices].toarray()
    return basis, np.array(basis_indices)


def extract_basis_sparse_svd(matrix, k=None):
    """
    Extract a basis for the column space of the given sparse matrix using SVD.
    
    Parameters:
    matrix (scipy.sparse.csr_matrix): The input sparse matrix.
    k (int): Number of singular values and vectors to compute. If None, compute the rank.
    
    Returns:
    numpy.ndarray: A matrix whose columns form a basis for the column space of the input matrix.
    numpy.ndarray: Indices of the columns that form the basis.
    """
    # convert to CSR format if not already in that format
    c_time = time.time()
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    
    c_time = timer(c_time,'convert to csr')
    # determine rank if not provided
    if k is None:
        k = min(matrix.shape) - 1
    
    # perform sparse SVD
    U, s, Vt = svds(matrix, k=k)
    c_time = timer(c_time,'svd')
    
    # extract the first 'rank' columns of U as the basis
    rank = np.sum(s > 1e-10)
    basis = U[:, :rank]
    c_time = timer(c_time,'basis')
    
    # find the indices of the basis columns
    # check which columns of the original matrix contribute to the basis
    basis_indices = []
    for i in tqdm(range(rank)):
        dot_products = np.abs(matrix.T @ basis[:, i])
        index = np.argmax(dot_products)
        basis_indices.append(index)
    
    return basis, np.array(basis_indices)


def extract_basis_jax_svd(matrix, k=None):
    """
    Extract a basis for the column space of the given sparse matrix using eigendecomposition in JAX.
    
    Parameters:
    matrix (jax sparse matrix): The input sparse matrix.
    k (int): Number of singular values and vectors to compute. If None, compute the rank.
    
    Returns:
    numpy.ndarray: A matrix whose columns form a basis for the column space of the input matrix.
    numpy.ndarray: Indices of the columns that form the basis.
    """
    from jax.experimental.sparse.linalg import lobpcg_standard
    N,M = matrix.shape 
    print(N,M)

    # convert to JAX sparse format if not already in that format
    c_time = time.time()
    if not isinstance(matrix, sparse.BCOO):
        matrix = sparse.BCOO(matrix, (N,M))

    c_time = timer(c_time,'convert to csr')
    # determine rank if not provided
    if k is None:
        k = min(N,M) - 1
    
    # perform sparse SVD
    rng = np.random.default_rng(0) # random seed

    AAT = lambda x: matrix @ matrix.T @ x
    AAT = matrix @ matrix.T
    initial = rng.normal(size=((N,5))).astype(jnp.float32)
    s, U = lobpcg_standard(AAT, initial, m=100, tol=1E-6)

    c_time = timer(c_time,'svd')

    # odentify the rank of the matrix (number of non-zero singular values)
    rank = jnp.sum(s > 1e-6)
    
    # extract the first 'rank' columns of U as the basis
    basis = U[:, :rank]
    c_time = timer(c_time,'basis')
    
    # find the indices of the basis columns
    # check which columns of the original matrix contribute to the basis
    basis_indices = []
    for i in tqdm(range(rank)):
        dot_products = jnp.abs(matrix.T @ basis[:, i])
        index = jnp.argmax(dot_products)
        basis_indices.append(index)
    
    return basis, jnp.array(basis_indices)


def normalize_dataset_jax(
    dataset: jnp.array,
    dtype: str,
    device: str,
    normalization_steps: int = 1,
    small_value: float = 1e-9):
    """Normalize the dataset and return the normalized dataset A and the transposed dataset B.

    Args:
        dataset (jnp array): The input dataset, samples as rows.
        normalization_steps (int, optional): The number of Sinkhorn normalization steps. For large numbers, we get bistochastic matrices. Defaults to 1 and should be larger or equal to 1.
        small_value (float): Small addition to the dataset to avoid numerical errors while computing OT distances. Defaults to 1e-6.

    Returns:
        Tuple[jnp arrays]: The normalized matrices A and B.
    """

    # Perform some sanity checks.
    assert len(dataset.shape) == 2  # correct shape
    assert jnp.sum(dataset < 0) == 0  # positivity
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    # Do a first normalization pass for A
    A = dataset / dataset.sum(1).reshape(-1, 1)
    A += small_value
    A /= A.sum(1).reshape(-1, 1)

    # Do a first normalization pass for B
    B = dataset.T / dataset.T.sum(1).reshape(-1, 1)
    B += small_value
    B /= B.sum(1).reshape(-1, 1)

    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        A, B = B.T / B.T.sum(1).reshape(-1, 1), A.T / A.T.sum(1).reshape(-1, 1)

    return A, B


def assert_nonzeros(dataset,labels1,labels2):
    """
    remove any rows or columns from the dataset that are all zeros
    """
    if jnp.sum(dataset.sum(1) == 0) == 0: # no row is 0
        dataset_new = dataset
        labels1_new = labels1
        labels2_new = labels2
    else:
        dataset_0 = jnp.argwhere(dataset.sum(1) == 0)
        dataset_new = jnp.delete(dataset,dataset_0,0)
        labels1_new = np.delete(labels1,dataset_0,0)
        labels2_new = np.delete(labels2,dataset_0,0)
    
    if jnp.sum(dataset_new.T.sum(1) == 0) == 0: # no column is 0
        dataset_2 = dataset_new
    else:
        dataset_0 = jnp.argwhere(dataset_new.T.sum(1) == 0)
        dataset_2 = jnp.delete(dataset_new.T,dataset_0,0)
        return dataset_2.T, labels1_new, labels2_new
    
    return dataset_2, labels1_new, labels2_new


def tree_init(A,B,K=[4,4],D=[12,12],tree='cluster',plotting=False,cdist=[None,None]):
    c_time = time.time()
    # construct trees
    btree = treeOT(B, method=tree, lam=0.001, n_slice=1, has_weights=False, k=K[0], d=D[0], custom_distance=cdist[0])
    atree = treeOT(A, method=tree, lam=0.001, n_slice=1, has_weights=False, k=K[1], d=D[1], custom_distance=cdist[1])
    
    # dense matrices and weight matrix shapes from tree
    a_Bsp = atree.Bsp
    b_Bsp = btree.Bsp
    a_z_dense = jnp.array(a_Bsp.todense(), dtype=bool).T
    b_z_dense = jnp.array(b_Bsp.todense(), dtype=bool).T
    
    del atree, btree

    # make sure no duplicate rows (if tree is set deeper than it should be for example)
    _, idx = np.unique(b_z_dense, axis=1, return_index=True)
    idx_sorted = np.sort(idx)
    
    # select the unique rows in the original order
    b_z_dense2 = b_z_dense[:,idx_sorted]

    if plotting == True:
        plt.imshow(b_z_dense2.T)
        plt.colorbar()
        plt.show()

    # and for the other tree
    _, idx = np.unique(a_z_dense, axis=1, return_index=True)
    idx_sorted = np.sort(idx)
    
    # select the unique rows in the original order
    a_z_dense2 = a_z_dense[:,idx_sorted]

    if plotting == True:
        plt.imshow(a_z_dense2.T)
        plt.colorbar()
        plt.show()

    c_time = timer(c_time,'tree initiation')
    
    return a_z_dense2, b_z_dense2


def sparse_construct(b_shape, b_z_dense, wvb_shape):
    """
    use sparse SVD method to return reduced rank matrix of tree distances
    Input shape of matrix (number leaves) and dense tree matrix
    Outputs reduced rank basis and upper triangular matrix of indices of full Y' (WxL^2)
    """
    # construct once-off matrices and tensors, report how long it takes
    c_time = time.time()
    b_up_tri = jnp.triu_indices(b_shape, k=1) # upper triangular of this
    # y_bll = (b_z_dense[:,None,:] + b_z_dense[None,:,:] - 2 * b_z_dense[:,None,:]*(b_z_dense[None,:,:])) # super large tensor ? can we make it smaller
    y_bll2 = jnp.logical_xor(b_z_dense[:,None,:], b_z_dense[None,:,:]).astype(bool)[b_up_tri].T
    nonzeros = y_bll2.nonzero()
    # argument pattern csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    # X_4 = sparse.csr_matrix((M[nonzeros], nonzeros), M.shape)
    c_time = timer(c_time,'y_bll')
    
    # Convert to CSR format if not already in that format
    c_time = time.time()
    if not isinstance((y_bll2.astype(jnp.float32)[nonzeros],nonzeros), csr_matrix):
        y_bll2 = csr_matrix((y_bll2.astype(jnp.float32)[nonzeros],nonzeros))
    c_time = timer(c_time,'convert to csr')
    del nonzeros
    
    # use sparse matrices to make more efficient (sparse basis set decomposition and report the ranks)
    y_b_red, ij_b = (extract_basis_sparse_svd(y_bll2,k=wvb_shape-1)) # can substitute extract_basis_jax_svd method
    c_time = timer(c_time,'extract basis 1')
    print('ybll2 red rank ',jnp.linalg.matrix_rank(y_b_red),' with shape ',y_bll2.shape, y_b_red.shape)
    y_b_red = y_b_red.T
    
    del y_bll2

    return y_b_red, b_up_tri
  

def tree_xor(B,node1,node2):
  """
  B is a tree matrix of size W x N_leaf where W is the number of nodes
  node1, node2 are the indices of the leaf nodes to be compared
  """
  return jnp.logical_xor(B[:,node1],B[:,node2]) 
  #return B[:,node1]+B[:,node2]-2*B[:,node1]*B[:,node2]


def basis_tree(B,basis=0,rank=0,indices=[],key = random.PRNGKey(0)):
  """
  Input: B, the tree matrix
  Recursively updates basis vector set and rank, then returns both
  When there are ties between leaves, always takes first
  """
  N,L = B.shape # number nodes and number leaves
  assert N > L

  # base cases
    
  if N == 3:
    # only one vector, compute and finish
    basis[rank] = (tree_xor(B,0,1))
    indices.append([0,1])
    rank += 1
    return basis, rank
  
  else:
    # select (smallest) subtree that is just adjacent leaves
    rowcts = jnp.sum(B[L:],axis=1)
    rowcts = jnp.where(rowcts>1,rowcts,L+1)
    min_value = rowcts.min()

    # select row with min count randomly 
    min_indices = jnp.argwhere(rowcts == min_value).flatten()  # Flatten to 1D
    key, subkey = random.split(key)
    curr_node = random.choice(key,min_indices)
    #curr_node = jnp.argmin(rowcts)
    curr_ct = int(rowcts[curr_node])
    # print(curr_node,curr_ct)      
    if curr_ct != L+1:
      curr_leaves = np.argwhere(B[L+curr_node])
      # print(curr_leaves)

      if curr_ct == 2: # include adjacent end-leaf pair and one other
        basis[rank] = (tree_xor(B,curr_leaves[0][0],curr_leaves[1][0]))
        indices.append([curr_leaves[0][0],curr_leaves[1][0]])
        rank += 1
        if curr_leaves[0][0]+1 != curr_leaves[1][0]:
          basis[rank] = (tree_xor(B,curr_leaves[0][0]+1,curr_leaves[1][0]))
          indices.append([curr_leaves[0][0]+1,curr_leaves[1][0]])
        elif curr_leaves[0][0] > 0:
          basis[rank] = (tree_xor(B,curr_leaves[0][0]-1,curr_leaves[1][0]))
          indices.append([curr_leaves[0][0]-1,curr_leaves[1][0]])
        else:
          basis[rank] = (tree_xor(B,curr_leaves[0][0]+2,curr_leaves[1][0]))
          indices.append([curr_leaves[0][0]+2,curr_leaves[1][0]])
        rank += 1
      
      else: # more than 2
        for i in range(curr_ct-1):
          basis[rank] = (tree_xor(B,curr_leaves[0][0],curr_leaves[i+1][0]))
          indices.append([curr_leaves[0][0],curr_leaves[i+1][0]])
          rank += 1
        basis[rank] = (tree_xor(B,curr_leaves[1][0],curr_leaves[2][0]))
        indices.append([curr_leaves[1][0],curr_leaves[2][0]])
        rank += 1
      for i in curr_leaves[1:]:
        B[:,i[0]] = 0
      basis, rank, indices  = basis_tree(B,basis,rank,indices)

    return basis, rank, indices
    

def basis_tree_rnd(B,basis=0,rank=0,indices=[],key = random.PRNGKey(0)):
  """
  Input: B, the tree matrix
  Recursively updates basis vector set and rank, then returns both
  Allows for some randomness when there are ties between leaves
  """
  N,L = B.shape # number nodes and number leaves
  assert N > L

  # base cases
    
  if N == 3:
    # only one vector, compute and finish
    basis[rank] = (tree_xor(B,0,1))
    indices.append([0,1])
    rank += 1
    return basis, rank
  
  else:
    # select (smallest) subtree that is just adjacent leaves
    rowcts = jnp.sum(B[L:],axis=1)
    rowcts = jnp.where(rowcts>1,rowcts,L+1)
    min_value = rowcts.min()

    # select row with min count randomly 
    min_indices = jnp.argwhere(rowcts == min_value).flatten()  # Flatten to 1D
    key, subkey = random.split(key)
    curr_node = random.choice(key,min_indices)

    curr_ct = int(rowcts[curr_node])
    # print(curr_node,curr_ct)      
    if curr_ct != L+1:
      curr_leaves = np.argwhere(B[L+curr_node])
      key, subkey = random.split(key)
      # choose which leaf to keep (connect to parent recursively) randomly
      ind_perm = random.permutation(key,curr_ct) # kept leaf is index left out
        
      if curr_ct == 2: # include adjacent end-leaf pair and one other
        basis[rank] = (tree_xor(B,curr_leaves[ind_perm[0]][0],curr_leaves[ind_perm[1]][0]))
        indices.append([curr_leaves[ind_perm[0]][0],curr_leaves[ind_perm[1]][0]])
        rank += 1
        if curr_leaves[ind_perm[0]][0] < curr_leaves[ind_perm[1]][0]:
            if curr_leaves[ind_perm[0]][0]+1 != curr_leaves[ind_perm[1]][0]:
              basis[rank] = (tree_xor(B,curr_leaves[ind_perm[0]][0]+1,curr_leaves[ind_perm[1]][0]))
              indices.append([curr_leaves[ind_perm[0]][0]+1,curr_leaves[ind_perm[1]][0]])
            elif curr_leaves[ind_perm[0]][0] > 0:
              basis[rank] = (tree_xor(B,curr_leaves[ind_perm[0]][0]-1,curr_leaves[ind_perm[1]][0]))
              indices.append([curr_leaves[ind_perm[0]][0]-1,curr_leaves[ind_perm[1]][0]])
            else:
              basis[rank] = (tree_xor(B,curr_leaves[ind_perm[0]][0]+2,curr_leaves[ind_perm[1]][0]))
              indices.append([curr_leaves[ind_perm[0]][0]+2,curr_leaves[ind_perm[1]][0]])
        else:
            if curr_leaves[ind_perm[0]][0]-1 != curr_leaves[ind_perm[1]][0]:
              basis[rank] = (tree_xor(B,curr_leaves[ind_perm[0]][0]-1,curr_leaves[ind_perm[1]][0]))
              indices.append([curr_leaves[ind_perm[0]][0]-1,curr_leaves[ind_perm[1]][0]])
            elif curr_leaves[ind_perm][0] < L-1:
              basis[rank] = (tree_xor(B,curr_leaves[ind_perm[0]][0]+1,curr_leaves[ind_perm[1]][0]))
              indices.append([curr_leaves[ind_perm[0]][0]+1,curr_leaves[ind_perm[1]][0]])
            else:
              basis[rank] = (tree_xor(B,curr_leaves[ind_perm[0]][0]-2,curr_leaves[ind_perm[1]][0]))
              indices.append([curr_leaves[ind_perm[0]][0]-2,curr_leaves[ind_perm[1]][0]])
        rank += 1
        B[:,curr_leaves[ind_perm[1]][0]] = 0
      
      else: # more than 2
        for i in range(curr_ct-1):
          basis[rank] = (tree_xor(B,curr_leaves[i][0],curr_leaves[i+1][0]))
          indices.append([curr_leaves[i][0],curr_leaves[i+1][0]])
          rank += 1
        basis[rank] = (tree_xor(B,curr_leaves[0][0],curr_leaves[-1][0]))
        indices.append([curr_leaves[0][0],curr_leaves[-1][0]])
        rank += 1

        # choose which leaf to leave out randomly
        ind_perm = random.permutation(key,curr_ct)
        for i in ind_perm[1:]:
            B[:,curr_leaves[i][0]] = 0
      basis, rank, indices  = basis_tree_rnd(B,basis,rank,indices)

    return basis, rank, indices


def basis_tree_legacy(B,basis=0,rank=0,indices=[]):
  """
  Input: B, the tree matrix
  Recursively updates basis vector set and rank, then returns both
  """
  N,L = B.shape # number nodes and number leaves
  assert N > L

  # base cases
    
  if N == 3:
    # only one vector, compute and finish
    basis[rank] = (tree_xor(B,0,1))
    indices.append([0,1])
    rank += 1
    return basis, rank
  
  else:
    # select (smallest) subtree that is just adjacent leaves
    rowcts = jnp.sum(B[L:],axis=1)
    rowcts = jnp.where(rowcts>1,rowcts,L+1)
    
    curr_node = jnp.argmin(rowcts)
    curr_ct = int(rowcts[curr_node])
    # print(curr_node,curr_ct)      
    if curr_ct != L+1:
      curr_leaves = np.argwhere(B[L+curr_node])
      # print(curr_leaves)

      if curr_ct == 2: # include adjacent end-leaf pair and one other
        basis[rank] = (tree_xor(B,curr_leaves[0][0],curr_leaves[1][0]))
        indices.append([curr_leaves[0][0],curr_leaves[1][0]])
        rank += 1
        if curr_leaves[0][0]+1 != curr_leaves[1][0]:
          basis[rank] = (tree_xor(B,curr_leaves[0][0]+1,curr_leaves[1][0]))
          indices.append([curr_leaves[0][0]+1,curr_leaves[1][0]])
        elif curr_leaves[0][0] > 0:
          basis[rank] = (tree_xor(B,curr_leaves[0][0]-1,curr_leaves[1][0]))
          indices.append([curr_leaves[0][0]-1,curr_leaves[1][0]])
        else:
          basis[rank] = (tree_xor(B,curr_leaves[0][0]+2,curr_leaves[1][0]))
          indices.append([curr_leaves[0][0]+2,curr_leaves[1][0]])
        rank += 1
      
      else: # more than 2
        for i in range(curr_ct-1):
          basis[rank] = (tree_xor(B,curr_leaves[0][0],curr_leaves[i+1][0]))
          indices.append([curr_leaves[0][0],curr_leaves[i+1][0]])
          rank += 1
        basis[rank] = (tree_xor(B,curr_leaves[1][0],curr_leaves[2][0]))
        indices.append([curr_leaves[1][0],curr_leaves[2][0]])
        rank += 1
      for i in curr_leaves[1:]:
        B[:,i[0]] = 0
      basis, rank, indices  = basis_tree_legacy(B,basis,rank,indices)

    return basis, rank, indices


def difference_subset(newa,ij_a,a_up_tri,b_z_dense,filename,save=True):
    """ 
    compute differences between feature vectors based on subset of indices selected
    returns Z
    """
    A_norm = newa#/jnp.sum(newa,axis=0)
    
    if a_up_tri.shape == newa.shape:
        A_diff_red = jnp.array([A_norm[p[0],:] - A_norm[p[1],:] for p in ij_a])
    else:
        A_diff_red = (A_norm[:,None,:]-A_norm[None,:,:])[a_up_tri][ij_a]

    z_ll = jnp.einsum('kn,in->ki', b_z_dense.T, A_diff_red)

    if save == True:
        filehandler = open(b"data/save/"+filename.encode('ascii')+b".obj","wb")
        pkl.dump({'z_ll':z_ll, 'tri':a_up_tri, 'z_ll':z_ll},filehandler)
    
    del A_diff_red, a_up_tri

    return z_ll 


def alt_twd(w,B,A):
    """
    returns the distance matrices for the tree using Jax vmap
    """
    n_samp = A.shape[0]
    distances = np.zeros((n_samp,n_samp))
    
    wB = jnp.diag(w) @ B.T
    print(wB.shape)
    for i in tqdm(range(n_samp)):
        distances[i] = vmap(lambda b: jnp.sum(jnp.abs(wB@(A[i]-b))))(A)
        #distances = vmap(lambda a: vmap(lambda b: jnp.sum(jnp.abs(wB@(a-b))))(A))(A)
    distances/=distances.max()
    return distances
