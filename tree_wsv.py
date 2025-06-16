import matplotlib.pyplot as plt
import numpy as np
from treeOT import *
import scipy.optimize as spo
from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
import time
import utils
from tqdm import tqdm
import scanpy as sc
import pickle as pkl
from scipy.sparse.linalg import svds
from sklearn.metrics import silhouette_score
import wsingular as wsingular
import torch
import sys
from scipy.io import loadmat
import umap

sys.setrecursionlimit(10000)

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", False)

# define the dtype and device to work with.
print('hi')
dtype = jnp.float32
device = "gpu"

## TODO: def tree_wsv():


# def difference_subset(newa,ij_a,a_up_tri,b_z_dense,filename,save=True):
#     """ 
#     compute differences between feature vectors based on subset of indices selected
#     returns Z
#     """
#     A_norm = newa/jnp.sum(newa,axis=0)
    
#     if a_up_tri.shape == newa.shape:
#         A_diff_red = jnp.array([A_norm[p[0],:] - A_norm[p[1],:] for p in ij_a])
#     else:
#         A_diff_red = (A_norm[:,None,:]-A_norm[None,:,:])[a_up_tri][ij_a]

#     z_ll = jnp.einsum('kn,in->ki', b_z_dense.T, A_diff_red)

#     if save == True:
#         filehandler = open(b"data/save/"+filename.encode('ascii')+b".obj","wb")
#         pkl.dump({'y_red':y_a_red, 'tri':a_up_tri, 'z_ll':z_ll},filehandler)
    
#     del A_diff_red, a_up_tri

#     return z_ll 

epsilon = 4e-15

# adata = sc.read_h5ad('data/pbmc_processed.h5ad')
# adata = sc.read_h5ad('data/tasic.h5ad')
# #dataset_pre = jnp.array(adata.X.todense())
# dataset_pre = jnp.array(adata.X)
# # print(adata.obs['broad_type'])
# # print(adata.obs['primary_type'])

# # key = jax.random.PRNGKey(0)
# # ind_perm = jax.random.permutation(key,dataset_pre.shape[0])[:1000]
# # dataset = dataset_pre[ind_perm,:]
# # dataset = dataset[:,ind_perm]
# dataset = np.asarray(dataset_pre)
# print(dataset.shape)
# dataset = utils.assert_nonzeros(dataset) # removes 0-rows and 0-columns
# print(dataset.shape)
# np_array = np.asarray(dataset)

# class = loadmat('/nfs/ghome/live/kdusterwald/Downloads/natimg2800_M170.mat')
# npx = loadmat('/nfs/ghome/live/kdusterwald/Downloads/stim_lab.mat')
# dataset = npx['stim']['resp'][0][0]

# filehandler = open(b"data/lung_proc_10x_2.obj","rb")
# b = pkl.load(filehandler)

# filehandler = open(b"data/save/ori32.obj",'rb')
# b = pkl.load(filehandler)

# dataset = b['X'][:,:9000]
# labels = b['labels'][:9000]

# dataset = jnp.array(b['X'].todense())
# print(dataset.shape)
# labels = np.array(b['l_1'])
# labels2 = np.array(b['l_2'])

# print(labels.shape,labels2.shape)

# dataset, labels, label2 = utils.assert_nonzeros(dataset,labels,labels2)

# numpy_arr = np.asarray(dataset)
# print(dataset.shape)

# torch_data = torch.from_numpy(numpy_arr).type(torch.DoubleTensor).cuda()

# sc.pp.pca(b)
# sc.pp.neighbors(b)
# sc.tl.umap(b)
# # sc.pl.umap(b, color='predicted.celltype.l1', save='dataset_cells.pdf')
# sc.pl.umap(b, color='ann_level_1', save='lung_cells.pdf')

# Define the dimensions of our problem -- toy data
n_samples = 1000
n_features = 1000

samp_arr = jnp.linspace(0,n_samples,n_samples,endpoint=False)
feat_arr = jnp.linspace(0,n_features,n_features,endpoint=False)

dataset = jnp.abs(samp_arr[:,None]/n_samples - feat_arr[None,:]/n_features % 1)

# Iterate over the features and samples.
dataset = np.zeros((n_samples, n_features), dtype=dtype)
for i in range(n_samples):
    for j in range(n_features):

        # Fill the dataset with translated histograms.
        dataset[i, j] = i/n_samples - j/n_features
        dataset[i, j] = abs(dataset[i, j] % 1)

# Take the distance to 0 on the torus.
dataset = jnp.fmin(dataset, 1 - dataset)

# Make it a gaussian.
dataset = jnp.exp(-(dataset)**2 / 0.1)

# define the dimensions of the problem
n_samples, n_features = dataset.shape
print(n_samples,n_features)

# samp_arr = jnp.linspace(0,n_samples,n_samples,endpoint=False)
# feat_arr = jnp.linspace(0,n_features,n_features,endpoint=False)

# dataset = jnp.abs(samp_arr[:,None]/n_samples - feat_arr[None,:]/n_features % 1)

# # Take the distance to 0 on the torus.
# dataset = jnp.fmin(dataset, 1 - dataset)

# # Make it a gaussian.
# dataset = jnp.exp(-(dataset)**2 / 0.1)

# np_array = np.asarray(dataset)
# torch_data = torch.from_numpy(np_array).type(torch.DoubleTensor).cuda()

# create datasets and normalise per row, column respectively
A, B = utils.normalize_dataset_jax(
            dataset,
            normalization_steps=1,
            small_value=1e-7,
            dtype=dtype,
            device=device,
)

#R_A = cdist(A, A, metric='minkowski',p=1)
#R_B = cdist(B, B, metric='minkowski',p=1)

newa = A #/ R_A.max()
newb = B #/ R_B.max()

#del R_A, R_B

n_leaf, d = dataset.shape
print('init done')

# filehandler = open(b"data/save/minimal_cells_lung10x.obj","rb")
# mc = pkl.load(filehandler)

# twd_a = utils.alt_twd(mc['gene_w'],mc['gene_B'],mc['cell_norm'])
# twd_b = utils.alt_twd(mc['cell_w'],mc['cell_B'],mc['gene_norm'])

# try:
#     print(silhouette(np.asarray(twd_a),labels))
#     twd_cell = twd_a
# except:
#     print(silhouette(np.asarray(twd_b),labels))
#     twd_cell= twd_b

# umap_model = umap.UMAP(metric='euclidean', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(dataset)

# df = labels
# categories = np.unique(df)

# colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

# fig = plt.figure(figsize=(8,6))
# for i, category in enumerate(categories):
#     subset = umap_model[df == category]
#     plt.scatter(subset[:,0], subset[:,1], label=category, color=colors[i], s=1)

# plt.legend(markerscale=5)
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')

# plt.savefig('lung_umap_euc.png',dpi=150)

# plt.close()

# umap_model = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(twd_cell)

# df = labels
# categories = np.unique(df)

# colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

# fig = plt.figure(figsize=(8,6))
# for i, category in enumerate(categories):
#     subset = umap_model[df == category]
#     plt.scatter(subset[:,0], subset[:,1], label=category, color=colors[i], s=1)

# plt.legend(markerscale=5)
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')

# plt.savefig('lung10x_umap_end.png',dpi=150)

# plt.close()


# umap_model = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(twd_cell)

# df = labels2
# categories = np.unique(df)

# colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

# fig = plt.figure(figsize=(8,6))
# for i, category in enumerate(categories):
#     subset = umap_model[df == category]
#     plt.scatter(subset[:,0], subset[:,1], label=category, color=colors[i], s=1)

# plt.legend(markerscale=5)
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')

# plt.savefig('lung10x_umap_end_labels2.png',dpi=150)

# plt.close()

# tree WD
start_tree = time.time()

# get tree parameter matrices
a_z_dense,b_z_dense = utils.tree_init(A,B,K=[3,3],D=[10,10],tree='cluster')#,cdist=[np.asarray(twd_a),np.asarray(twd_b)])
a_shape,wva_shape = a_z_dense.shape
b_shape,wvb_shape = b_z_dense.shape

sil_old = -0.1
twd_a_in, twd_b_in = [None,None]
best = 0

for i in range(1):
    # get basis vectors for tree parameter matrices
    c_time = time.time()
    print('extracting basis for tree')
    y_b_red,rank_b,ij_b = utils.basis_tree_rnd(np.array(b_z_dense.T),basis=np.zeros((wvb_shape-1,wvb_shape)).astype(jnp.bool),rank=0,indices=[])
    print('B rank',jnp.linalg.matrix_rank(y_b_red,tol=1e-5),'reported',rank_b,'with shape (',y_b_red.shape,')')
    while y_b_red.shape[0] > rank_b:
        y_b_red = y_b_red[:-1,:]
    y_a_red,rank_a,ij_a = utils.basis_tree_rnd(np.array(a_z_dense.T),basis=np.zeros((wva_shape-1,wva_shape)).astype(jnp.bool),rank=0,indices=[])
    print('A rank',jnp.linalg.matrix_rank(y_a_red,tol=1e-5),'reported',rank_a,'with shape (',y_a_red.shape,')')
    while y_a_red.shape[0] > rank_a:
        y_a_red = y_a_red[:-1,:]#
    breakpoint()
    c_time = utils.timer(c_time,'extract basis x2')
    
    # create once off matrices that include all pairings of vectors, a short Z matrix
    print('computing difference matrices based on basis indices')
    z_bll = utils.difference_subset(newa,ij_a,jnp.zeros_like(newa),b_z_dense,'atree_lung',save=True)
    z_all = utils.difference_subset(newb,ij_b,jnp.zeros_like(newb),a_z_dense,'btree_lung',save=True)
    c_time = utils.timer(c_time,'short z matrices, checkpoints')
    
    # loops
    n_inner_iter = 20
    
    # initiation for iterative steps
    converge = np.zeros((2,n_inner_iter), dtype=dtype)
    wva = jnp.array(np.random.rand(wva_shape), dtype=dtype)
    wvb = jnp.array(np.random.rand(wvb_shape), dtype=dtype)
    total_converge = 1

    loss = np.zeros((2,n_inner_iter), dtype=dtype)
    
    for k in tqdm(range(n_inner_iter)):
        if total_converge > epsilon:
            # first for B
            if k == 0:
                twd_b_2 = jnp.ones(jnp.sum(jnp.abs(jnp.einsum('i,ij->ij',wvb.T.squeeze(),z_bll)),0).shape)
            wva_temp = twd_b_2/twd_b_2.max()
            twd_b_2 = jnp.sum(jnp.abs(jnp.einsum('i,ij->ij',wvb.T.squeeze(),z_bll)),0) # Wasserstein distances
            twd_b_phi2 = (twd_b_2 / twd_b_2.max()).ravel() # SV normalisation # make into vector 
            #wva_all = spo.nnls(y_a_red,twd_b_phi2) # solve system of linear equations using NNLS
            wva_all = jnp.linalg.lstsq(y_a_red, twd_b_phi2)
            converge[1,k] = np.linalg.norm(wva-jnp.array(wva_all[0], dtype=dtype))
            wva = jnp.array(wva_all[0], dtype=dtype)

            loss[1,k] = utils.hilbert_distance_jax(wva_temp,twd_b_2)
            
            # then for A
            if k == 0:
                twd_a_2 = jnp.ones(jnp.sum(jnp.abs(jnp.einsum('i,ij->ij',wva.T.squeeze(),z_all)),0).shape)
            wvb_temp = twd_a_2/twd_a_2.max()
            twd_a_2 = jnp.sum(jnp.abs(jnp.einsum('i,ij->ij',wva.T.squeeze(),z_all)),0)
            twd_a_phi2 = (twd_a_2 / twd_a_2.max()).ravel()
            #wvb_all = spo.nnls(y_b_red,twd_a_phi2)
            wvb_all = jnp.linalg.lstsq(y_b_red, twd_a_phi2)
            converge[0,k] = np.linalg.norm(wvb-jnp.array(wvb_all[0], dtype=dtype))
            wvb = jnp.array(wvb_all[0], dtype=dtype)

            loss[0,k] = utils.hilbert_distance_jax(wvb_temp,twd_a_2)
        total_converge = converge[0,k]+converge[1,k]

    plt.plot(jnp.log(loss[0]),label='$\mathcal{W}_{\mathcal{T}_A}$')
    plt.plot(jnp.log(loss[1]),label = '$\mathcal{W}_{\mathcal{T}_B}$')
    plt.ylabel('$\log(d_H)$')
    plt.xlabel('Iterations')
    plt.xticks(range(0,19),range(1,20))
    plt.legend()
    plt.savefig('Convergence.png',dpi=150)
    plt.show()

    breakpoint()
    
    print('time WSV tree: ' + str(time.time() - start_tree))
    
    print(converge)
    
    A_norm = newa/jnp.sum(newa,axis=0)
    B_norm = newb/jnp.sum(newb,axis=0)
    
    filehandler = open(b"data/save/minimal_cells_lung12x.obj","wb")
    pkl.dump({'gene_B':a_z_dense, 'cell_B':b_z_dense, 'gene_norm':A_norm, 'cell_norm':B_norm,
             'gene_w':wva, 'cell_w':wvb},filehandler)
    
    twd_a = utils.alt_twd(wva,a_z_dense,B_norm)
    # plt.imshow(twd_a)
    # plt.savefig('images/twd_a_tasic.png')
    # plt.close()
    
    twd_b = utils.alt_twd(wvb,b_z_dense,A_norm)
    # plt.imshow(twd_b)
    # plt.savefig('images/twd_b_tasic.png')
    # plt.close()
    
    # sil = silhouette(twd_b,adata.obs['predicted.celltype.l1'])
    # sil2 = silhouette(twd_b,adata.obs['predicted.celltype.l2'])
    try:
        sil = utils.silhouette(twd_a,labels)
    except:
        sil = utils.silhouette(twd_b,labels)
    print('Iter',i,'ASW:',sil,'Best:',sil_old,'at iter',best)
    if sil > sil_old:
        best = i
        filehandler = open(b"data/save/gc_lung12x.obj","wb")
        pkl.dump({'gene':A,'gene_cost':twd_a,'cell':B,'cell_cost':twd_b},filehandler)
        sil_old = sil
    try:
        sil = utils.silhouette(twd_a,labels2)
    except:
        sil = utils.silhouette(twd_b,labels2)
    print('Level 2! Iter',i,'ASW:',sil)
    # get tree parameter matrices
    a_z_dense,b_z_dense = utils.tree_init(A,B,K=[3,3],D=[14,14],tree='cluster',cdist=[twd_a,twd_b])
    a_shape,wva_shape = a_z_dense.shape
    b_shape,wvb_shape = b_z_dense.shape
    
    umap_model = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(twd_b)
    
    df = labels
    categories = np.unique(df)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    
    fig = plt.figure(figsize=(8,6))
    for i, category in enumerate(categories):
        subset = umap_model[df == category]
        plt.scatter(subset[:,0], subset[:,1], label=category, color=colors[i], s=1)
    
    plt.legend(markerscale=5)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    
    plt.savefig('lung12x_umap_end.png',dpi=150)
    
    plt.close()
    
    
    umap_model = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(twd_b)
    
    df = labels2
    categories = np.unique(df)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    
    fig = plt.figure(figsize=(8,6))
    for i, category in enumerate(categories):
        subset = umap_model[df == category]
        plt.scatter(subset[:,0], subset[:,1], label=category, color=colors[i], s=1)
    
    plt.legend(markerscale=5)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    
    plt.savefig('lung12x_umap_end_labels2.png',dpi=150)
    
    plt.close()

breakpoint()
# Sinkhorn
start_ssv = time.time()
# Compute the SSV.
C, D = wsingular.sinkhorn_singular_vectors(
    torch_data,
    eps=0.1,
    dtype=torch.double,
    device=device,
    n_iter=15,
    progress_bar=True,
    tau=0.001
)

print('time SSV: ' + str(time.time() - start_ssv))

C_true = C
D_true = D

filehandler = open(b"data/save/lung10x_wsv.obj","wb")
pkl.dump({'gene_cost':C_true,'cell_cost':D_true},filehandler)

print('Hilbert diff')
print(np.linalg.norm(twd_a-np.asarray(C_true.cpu())))
print(np.linalg.norm(twd_b-np.asarray(D_true.cpu())))

print('sil true')
print(utils.silhouette(np.asarray(D_true.cpu()),b['l_1']))
print(utis.silhouette(np.asarray(D_true.cpu()),b['l_2']))

# plain WSV
start_wsv = time.time()
# Compute the WSV.
C, D = wsingular.wasserstein_singular_vectors(
    torch_data,
    n_iter=10,
    dtype=torch.double,
    device=device,
    tau = 0.1,
    progress_bar=True
)

print('time WSV: ' + str(time.time() - start_wsv))
C_true = C
D_true = D

filehandler = open(b"data/save/full_gene_wsv2.obj","wb")
pkl.dump({'gene_cost':C_true,'cell_cost':D_true},filehandler)
         
print('Hilbert diff')
print(np.linalg.norm(twd_a-np.asarray(C_true.cpu())))
print(np.linalg.norm(twd_b-np.asarray(D_true.cpu())))

print('sil true')
print(utils.silhouette(np.asarray(D_true.cpu()),adata.obs['predicted.celltype.l1']))
print(utils.silhouette(np.asarray(D_true.cpu()),adata.obs['predicted.celltype.l2']))
print(utils.silhouette(np.asarray(D_true.cpu()),adata.obs['predicted.celltype.l3']))

breakpoint()

# Display the SSV.

# z_bll_old = jnp.einsum('kn,i,nj->kij', b_z_dense.T, A_norm, A_norm)

# twd_b_2 = jnp.sum(jnp.abs(jnp.einsum('i,ijk->ijk', wvb.T.squeeze(), z_bll_old)), 0)
# twd_b_phi = twd_b_2 / twd_b_2.max()

# z_all_old = jnp.einsum('kn,i,nj->kij', a_z_dense.T, B_norm, B_norm)

# twd_a_2 = jnp.sum(jnp.abs(jnp.einsum('i,ijk->ijk', wva.T.squeeze(), z_all_old)), 0)
# twd_a_phi = twd_a_2 / twd_a_2.max()

# filehandler = open(b"data/save/genecell_distances.obj","wb")
# pkl.dump({'gene_twd':twd_a_phi, 'cell_twd':twd_b_phi, 'gene_w':wva, 'cell_w':wvb, 
#           'gene_tree':a_z_dense, 'cell_tree': b_z_dense},filehandler)

# print(silhouette(twd_b_phi,adata.obs['predicted.celltype.l1']))
# print(silhouette(twd_b_phi,adata.obs['predicted.celltype.l2']))

# A_norm = newa/jnp.sum(newa,axis=0)
# A_diff = A_norm[:,None,:]-A_norm[None,:,:]
# B_norm = newb/jnp.sum(newb,axis=0)
# B_diff = B_norm[:,None,:]-B_norm[None,:,:]

# z_bll_old = jnp.einsum('kn,ijn->kij', b_z_dense.T, A_diff)
# twd_b_2 = jnp.sum(jnp.abs(jnp.einsum('i,ijk->ijk',wvb.T.squeeze(),z_bll_old)),0)
# twd_b_phi = twd_b_2 / twd_b_2.max()

# z_all_old = jnp.einsum('kn,ijn->kij', a_z_dense.T, B_diff)
# twd_a_2 = jnp.sum(jnp.abs(jnp.einsum('i,ijk->ijk',wva.T.squeeze(),z_all_old)),0)
# twd_a_phi = twd_a_2 / twd_a_2.max()

# utils.display_cost(twd_a_phi,twd_b_phi,n_samples,n_features,name='cell_gene')
# plt.show()

breakpoint()

# plain WSV
start_wsv = time.time()
# Compute the WSV.
C, D = wsingular.wasserstein_singular_vectors(
    dataset,
    n_iter=100,
    dtype=dtype,
    device=device,
    tau = 0.1
)

print('time WSV: ' + str(time.time() - start_wsv))


breakpoint()

#### STOP HERE: it is unlikely that we will be able to run the next step, since the ann_data has changed and no longer has these labels
print(utils.silhouette(D,adata.obs['predicted.celltype.l1']))
print(utils.silhouette(D,adata.obs['predicted.celltype.l2']))

utils.display_cost(C,D,n_samples,n_features,name='cluster_cost_twdsv')