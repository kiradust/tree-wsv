import umap.umap_ as umap
from umap.umap_ import UMAP
import numpy as np
import importlib
from helper import import_data
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from tree_wsv import tree_wsv
import jax.numpy as jnp
import pickle as pkl

idx, X_import, data, X_stack, idx_stack, unique_vals, X_agg, C_agg, C_cross = import_data()

# angles = unique_vals
angles = idx

# naive umaps
umap_model = UMAP().fit(data)
print(data.shape)
# umap.plot.points(umap_model, labels=angles, cmap='plasma', show_legend=False)

embedding = UMAP(n_components=3).fit_transform(data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=angles, cmap='plasma')
fig.colorbar(sc, ax=ax)
plt.savefig('image/naive_umap_3d.png', dpi=150)
plt.show()

embedding = UMAP(n_components=2).fit_transform(data)
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                c=angles, cmap='plasma')
fig.colorbar(sc, ax=ax)
plt.savefig('image/naive_umap_2d.png', dpi=150)
plt.show()

# unsup gw learning
a_z_dense, b_z_dense, twd_a, twd_b = tree_wsv(jnp.array(data[:,:2000]), n_iter=2)

print(twd_a.shape)
print(twd_b.shape)

# save
filehandler = open(b"data/save/ali_data.obj","wb")
pkl.dump({'stim_m':a_z_dense, 'neur_m':b_z_dense, 'stim_d':twd_a, 'neur_d':twd_b},filehandler)

# fit umaps
umap_model = UMAP().fit(data)
print(data.shape)
# umap.plot.points(umap_model, labels=angles, cmap='plasma', show_legend=False)

embedding = UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=3, random_state=42).fit_transform(twd_b)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=angles, cmap='plasma')
fig.colorbar(sc, ax=ax)
plt.savefig('image/fugm_umap_3d.png', dpi=150)
plt.show()

embedding = UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(twd_b)
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                c=angles, cmap='plasma')
fig.colorbar(sc, ax=ax)
plt.savefig('image/fugm_umap_2d.png', dpi=150)
plt.show()


embedding = UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=3, random_state=42).fit_transform(twd_a)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=angles, cmap='plasma')
fig.colorbar(sc, ax=ax)
plt.savefig('image/fugm_a_umap_3d.png', dpi=150)
plt.show()

embedding = UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(twd_a)
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                c=angles, cmap='plasma')
fig.colorbar(sc, ax=ax)
plt.savefig('image/fugm_a_umap_2d.png', dpi=150)
plt.show()