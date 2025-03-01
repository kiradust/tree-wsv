import wsingular as wsingular
import matplotlib.pyplot as plt
import numpy as np
from treeOT import *
import scipy.optimize as spo
import torch
import time
import utils
from tqdm import tqdm

# define the dtype and device
dtype = torch.double
device = "cpu"

# define the dimensions of the problem
n_samples = 80
n_features = 60
n_iter = 100

# initialize an empty dataset
dataset = torch.zeros((n_samples, n_features), dtype=dtype)
dataset_trans = torch.zeros((n_samples, n_features), dtype=dtype)

# iterate over the features and samples
for i in range(n_samples):
    for j in range(n_features):

        # fill the dataset with translated histograms
        dataset[i, j] = i/n_samples - j/n_features
        dataset[i, j] = torch.abs(dataset[i, j] % 1)

        # alternative (translated histogram)
        dataset_trans[i, j] = i/n_samples - j/n_features
        dataset_trans[i, j] = torch.abs((dataset_trans[i, j]%  1)+0.5)

# take the distance to 0 on the torus
dataset_trans = torch.min(dataset_trans, 1-dataset_trans)
dataset = torch.min(dataset, 1 - dataset)

# make it a guassian.
dataset = torch.exp(-(dataset**2) / 0.01)
dataset_trans = torch.exp(-(dataset_trans**2) / 0.01)
dataset = (dataset + 0.5 * dataset_trans)

# shuffle dataset for tree iterations
ind_perm = torch.randperm(dataset.size(0))
dataset = (dataset + 0.5 * dataset_trans)[ind_perm]

ind_perm = torch.randperm(dataset.size(1))
dataset = (dataset + 0.5 * dataset_trans)[:,ind_perm]

# plot the dataset
plt.title('The dataset')
plt.imshow(dataset)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

A, B = utils.normalize_dataset(
            dataset,
            normalization_steps=1,
            small_value=1e-5,
            dtype=dtype,
            device=device,
)

R_A = utils.regularization_matrix(A, p=1, dtype=dtype, device=device)
R_B = utils.regularization_matrix(B, p=1, dtype=dtype, device=device)

newa = A / R_A.max()
newb = B / R_B.max()
n_leaf, d = dataset.shape

C,D = torch.tensor(np.ones((n_features, n_features))), torch.tensor(np.ones((n_samples,n_samples)))

# plain WSV
start_wsv = time.time()
# Compute the WSV
C, D = wsingular.wasserstein_singular_vectors(
    dataset,
    n_iter=n_iter,
    dtype=dtype,
    device=device,
    tau = 0.1,
    progress_bar=True
)

print('time WSV: ' + str(time.time() - start_wsv))

# display the WSV
utils.display_cost(C,D,n_samples,n_features)

C_true = C
D_true = D

# tree WD
start_tree = time.time()

# construct trees -- k modifies the number of children
btree = treeOT(B, method='cluster', lam=0.001, n_slice=1, is_sparse=False, has_weights=False, k=3, d=4)
atree = treeOT(A, method='cluster', lam=0.001, n_slice=1, is_sparse=False, has_weights=False, k=3, d=4)

# construct once-off matrices and tensors
a_z_dense = torch.tensor(atree.Bsp.todense(), dtype=dtype).T
b_z_dense = torch.tensor(btree.Bsp.todense(), dtype=dtype).T
a_shape,wva_shape = a_z_dense.shape
b_shape, wvb_shape = b_z_dense.shape

c_time = time.time()
y_all = (a_z_dense[:,None,:] + a_z_dense[None,:,:] - 2 * a_z_dense[:,None,:]*(a_z_dense[None,:,:]))
y_all2 = y_all[np.triu_indices(a_shape, k=1)].T
y_bll = (b_z_dense[:,None,:] + b_z_dense[None,:,:] - 2 * b_z_dense[:,None,:]*(b_z_dense[None,:,:]))
y_bll2 = y_bll[np.triu_indices(b_shape, k=1)].T
c_time = utils.timer(c_time,'y_all and y_bll dense')

y_a_red,ij_a = (utils.extract_basis_sparse_svd(y_all2,k=wva_shape-1))
c_time = utils.timer(c_time,'extract basis 1')
y_b_red, ij_b = (utils.extract_basis_sparse_svd(y_bll2,k=wvb_shape-1))
c_time = utils.timer(c_time,'extract basis 2')
print('ybll2 red rank ',np.linalg.matrix_rank(y_b_red),' with shape ',y_bll2.shape, y_b_red.shape)
print('yall2 red rank ',np.linalg.matrix_rank(y_a_red),' with shape ',y_all2.shape, y_a_red.shape)
c_time = utils.timer(c_time,'ranks')
y_a_red = y_a_red.T
y_b_red = y_b_red.T

a_z_dense = torch.tensor(atree.Bsp.todense(), dtype=dtype).T
b_z_dense = torch.tensor(btree.Bsp.todense(), dtype=dtype).T

A_norm = newa/torch.sum(newa,axis=0)
A_diff = A_norm[:,None,:]-A_norm[None,:,:]
A_diff_red = A_diff[np.triu_indices(a_shape, k=1)][ij_a]
z_bll = torch.einsum('kn,in->ki', b_z_dense.T, A_diff_red)
B_norm = newb/torch.sum(newb,axis=0)
B_diff = (B_norm[:,None,:]-B_norm[None,:,:])
B_diff_red = B_diff[np.triu_indices(b_shape, k=1)][ij_b]
z_all = torch.einsum('kn,in->ki', a_z_dense.T, B_diff_red)

# loops
n_inner_iter = n_iter

loss_c = torch.zeros(n_iter, dtype=dtype)
loss_d = torch.zeros(n_iter, dtype=dtype)
wva = torch.tensor(np.random.rand(wva_shape), dtype=dtype)
wvb = torch.tensor(np.random.rand(wvb_shape), dtype=dtype)

for k in tqdm(range(n_inner_iter)):
    # first for B
    twd_b_2 = torch.sum(torch.abs(torch.einsum('i,ij->ij',wvb.T.squeeze(),z_bll)),0)
    twd_b_phi = twd_b_2 / twd_b_2.max()
    twd_b_phi2 = twd_b_phi.ravel()
    wva_all = spo.nnls(y_a_red,twd_b_phi2)
    wva = torch.tensor(wva_all[0],dtype=dtype)
    
    # then for A
    twd_a_2 = torch.sum(torch.abs(torch.einsum('i,ij->ij',wva.T.squeeze(),z_all)),0)
    twd_a_phi = twd_a_2 / twd_a_2.max()
    twd_a_phi2 = twd_a_phi.ravel()
    wvb_all = spo.nnls(y_b_red,twd_a_phi2)
    wvb = torch.tensor(wvb_all[0],dtype=dtype)

print('time WSV-many tree: ' + str(time.time() - start_tree))

# display
z_bll_old = torch.einsum('kn,ijn->kij', b_z_dense.T, A_diff)
twd_b_2 = torch.sum(torch.abs(torch.einsum('i,ijk->ijk',wvb.T.squeeze(),z_bll_old)),0)
twd_b_phi = twd_b_2 / twd_b_2.max()

z_all_old = torch.einsum('kn,ijn->kij', a_z_dense.T, B_diff)
twd_a_2 = torch.sum(torch.abs(torch.einsum('i,ijk->ijk',wva.T.squeeze(),z_all_old)),0)
twd_a_phi = twd_a_2 / twd_a_2.max()
# utils.display_cost(twd_a_phi,twd_b_phi,n_samples,n_features)

# print(np.linalg.norm(twd_a_phi-C_true))
# print(np.linalg.norm(twd_b_phi-D_true))
print('Hilbert distance cost 1:', utils.hilbert_distance(C_true,twd_a_phi))
print('Hilbert distance cost 2:', utils.hilbert_distance(D_true,twd_b_phi))
