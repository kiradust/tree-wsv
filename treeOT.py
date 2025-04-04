import numpy as np
from treelib import Tree
import copy
from tqdm import tqdm
import random
from scipy import sparse

class treeOT():
    def __init__(self, X, method='cluster', lam=0.0001, nmax=100000, k=5, d=6, n_slice=1, 
                 debug_mode=False, has_weights=True,  custom_distance=None):
        """
        Modified from Yamada et al., 2022 (https://github.com/oist/treeOT?tab=readme-ov-file) to allow custom distance inputs
         Parameter
         ----------
         X :
            a set of supports
         method :
            'cluster' (clustering tree) or 'quad' (quadtree)
         k : int
            a number of child nodes
         d : int
            depth of a tree
         n_slice : int
            the number of sampled trees
         lam: float
            the regularization parameter of Lasso
         nmax: int
            the number of training samples for Lasso
         has_weights: bool
            whether or not to calculate weights or input own
        custom_distance: matrix or None
            input a custom distance matrix for use with ClusterTree (otherwise, default to Euclidean distance)
         """

        self.n_slice = n_slice
        self.custom_distance = custom_distance

        for i in tqdm(range(n_slice)):

            if method=='quad': #Quadtree
                np.random.seed(i)
                tree = self.build_quadtree(X, random_shift=True, width=None, origin=None)
                print("build done")

            else: #Clustering tree
                random.seed(i)
                tree = self.build_clustertree(X, k, d, debug_mode=debug_mode)
                print("build done")

            self.Bsp = self.get_B_matrix(tree,X)

            if has_weights:
                
                wv = self.calc_weight(X,self.Bsp,lam=lam,nmax=nmax)

                if i == 0:
                    wB = self.Bsp.multiply(wv)
                else:
                    wB = sparse.vstack([wB,self.Bsp.multiply(wv)])

            else:
                    wB = np.zeros(X.shape)
                    self.B_dense = np.array(self.Bsp.todense())
                    self.y_all = self.calc_yall()

                    wv = np.zeros(self.B_dense.shape[0])

        self.wB = wB
        self.wv = wv


    def incremental_farthest_search(self, points, remaining_set, k, debug_mode=False):
        """ Modified for custom distance function """
        n_points = len(remaining_set)
        remaining_set = copy.deepcopy(remaining_set)

        if not debug_mode:
            # random.seed(0)
            solution_set = [remaining_set[random.randint(0, n_points - 1)]]
        else:
            solution_set = [remaining_set[0]]
        remaining_set.remove(solution_set[0])

        for i in range(k - 1):

            distance_list = []

            for idx in remaining_set:
                if self.custom_distance is not None:
                    in_distance_list = [self.custom_distance[idx, sol_idx] for sol_idx in solution_set]
                else:
                    in_distance_list = [self.distance(points[idx], points[sol_idx]) for sol_idx in solution_set] 
                distance_list.append(min(in_distance_list))

            sol_idx = remaining_set[np.argmax(distance_list)]
            remaining_set.remove(sol_idx)
            solution_set.append(sol_idx)

        return solution_set


    def distance(self, A, B):
        return np.linalg.norm(A - B)


    def grouping(self, points, remaining_set, solution_set):
        """ Modified for custom distance function """
        np.random.seed(0)
        n_points = len(points)
        remaining_set = copy.deepcopy(remaining_set)

        group = []
        for _ in range(len(solution_set)):
            group.append([])

        for idx in remaining_set:
            if self.custom_distance is not None:
                distance_list = [self.custom_distance[idx, sol_idx] for sol_idx in solution_set]
            else:
                distance_list = [self.distance(points[idx], points[sol_idx]) for sol_idx in solution_set]
           
            group_idx = np.argmin(distance_list)
            group[group_idx].append(idx)

        return group

    def clustering(self, points, remaining_set, k, debug_mode=False):
        solution_set = self.incremental_farthest_search(points, remaining_set, k, debug_mode=debug_mode)
        return self.grouping(points, remaining_set, solution_set)


    def _build_clustertree(self, X, remaining_set, k, d, debug_mode=False):
        tree = Tree()
        tree.create_node(data=None)

        if len(remaining_set) <= k or d == 1:
            for idx in remaining_set:
                tree.create_node(parent=tree.root, data=idx)
            return tree

        groups = self.clustering(X, remaining_set, k, debug_mode=debug_mode)
        # print(groups)
        for group in groups:
            if len(group) == 1:
                tree.create_node(parent=tree.root, data=group[0])
            else:
                subtree = self._build_clustertree(X, group, k, d - 1, debug_mode=debug_mode)
                tree.paste(tree.root, subtree)
        return tree


    def build_clustertree(self, X, k, d, debug_mode=False):
        """
        k : the number of child nodes
        d : the depth of the tree
        """
        remaining_set = [i for i in range(len(X))]
        return self._build_clustertree(X, remaining_set, k, d, debug_mode=debug_mode)


    def _build_quadtree(self, X, origin, remaining_idx, width):
        d = X.shape[1]  # dimension
        m = len(remaining_idx)  # number of samples (i.e., support size)

        tree = Tree()
        tree.create_node(data=None)

        loc = np.zeros(m).tolist()

        # divide the hypercube, and obtain which hypercube a point belong to.
        for i in range(len(remaining_idx)):
            for j in range(d):
                if X[remaining_idx[i]][j] > origin[j]:
                    loc[i] += 2 ** j

        child = list(set(loc))
        child_set = [[] for _ in range(len(child))]
        origin_set = []

        for i in range(len(child)):
            new_origin = np.zeros_like(origin)
            for j in range(d):
                if int(child[i]) & (2 ** j) != 0:
                    new_origin[j] = copy.deepcopy(origin[j]) + width / 2.0
                else:
                    new_origin[j] = copy.deepcopy(origin[j]) - width / 2.0
            origin_set.append(new_origin)

        for i in range(m):
            child_set[child.index(loc[i])].append(remaining_idx[i])

        for i in range(len(child)):
            if len(child_set[i]) == 1:
                tree.create_node(parent=tree.root, data=child_set[i][0])
            else:
                subtree = self._build_quadtree(X, origin_set[i], child_set[i], width / 2.0)
                tree.paste(tree.root, subtree)

        return tree


    def build_quadtree(self, X, random_shift=True, width=None, origin=None):
        """
        Assume that X[i] in [0, width]^d.
        """
        #np.random.seed(0)
        if random_shift:
            # check the assumption.
            if np.min(X) < 0:
                print("Warn : Assumption")
                X = X - np.min(X)
            elif np.min(X) != 0:
                print("Warn : Assumption")

            width = np.max(X)
            origin = np.random.uniform(low=0.0, high=width, size=X.shape[1])

        remaining_idx = [i for i in range(X.shape[0])]

        return self._build_quadtree(X, origin, remaining_idx, width)


    def get_B_matrix(self, tree, X):
        """Small mods for custom distance and matrix rank algorithm compatibility"""
        n_node = len(tree.all_nodes())
        n_leaf = X.shape[0]
        n_in   = n_node - n_leaf

        #B = np.zeros((n_node,n_leaf))

        in_node   = [node.identifier for node in tree.all_nodes() if node.data == None]
        in_node_index = [ii for ii in range(n_in)]
        leaf_node = [node.identifier for node in tree.all_nodes() if node.data != None]
        leaf_node_index = [node.data for node in tree.all_nodes() if node.data != None]
        path_leaves = tree.paths_to_leaves()

        n_edge = 0
        for path in path_leaves:
            n_edge += len(path)
        col_ind = np.zeros(n_edge)
        row_ind = np.zeros(n_edge)
        cnt = 0

        for path in path_leaves:
            # check node is leaf or not
            try:
                leaf_index = leaf_node_index[leaf_node.index(path[-1])]
                #B[leaf_index,leaf_index] = 1.0
                col_ind[cnt] = leaf_index
                row_ind[cnt] = leaf_index
                cnt += 1
                for node in path[:-1]:
                    in_index = in_node_index[in_node.index(node)] + n_leaf
                    #B[in_index,leaf_index] = 1.0
                    col_ind[cnt] = leaf_index
                    row_ind[cnt] = in_index
                    cnt+=1
            except ValueError:
                continue

        B = sparse.csc_matrix((np.ones(n_edge), (row_ind, col_ind)), shape=(n_node, n_leaf), dtype='float32')
        for i in range(n_node):
            if i < n_leaf:
                B[i,i] = 1
        return B


    def calc_weight(self, X, B, lam=0.001, seed=0, nmax=100000):
        """We do not really need this function -- modified for compatibility"""
        n_leaf, d = X.shape
        random.seed(seed)

        # Create B matrix
        #n_in = self.D2.shape[0]
        #B1 = np.linalg.solve(np.eye(n_in) - self.D1, self.D2)
        #B = np.concatenate((B1, np.eye(n_leaf)))

        dz = B.shape[0]

        np.random.seed(seed)
        ind1 = np.random.randint(0, n_leaf, nmax)
        ind2 = np.random.randint(0, n_leaf, nmax)

        c_all = np.zeros((nmax, 1))
        Z_all = np.zeros((dz, nmax))

        for ii in range(nmax):
            c_all[ii] = np.linalg.norm(X[ind1[ii], :] - X[ind2[ii], :], ord=2)
            Z_all[:, ii] = B[:, ind1[ii]] + B[:, ind2[ii]] - 2 * (B[:, ind1[ii]] * B[:, ind2[ii]])

        n_sample = nmax
        c = np.asfortranarray(c_all[:n_sample, 0].reshape((n_sample, 1)), dtype='float32')
        Z = np.asfortranarray(Z_all[:, :n_sample].transpose(), dtype='float32')

        # Solving nonnegative Lasso
        param = {'numThreads': -1, 'verbose': True,
                 'lambda1': lam, 'it0': 10, 'max_it': 2000, 'tol': 1e-3, 'intercept': False,
                 'pos': True}

        param['loss'] = 'square'
        param['regul'] = 'l1'

        W0 = np.zeros((Z.shape[1], c.shape[1]), dtype='float32', order="F")

        return W0

    def calc_yall(self):
        # compute the matrix for all pairwise leaf-to-leaf distances
        y_all = (self.B_dense[:,None,:] + self.B_dense[:,:,None] - 2 * (self.B_dense[:,None,:]*(self.B_dense[:,:,None])))
        return y_all
