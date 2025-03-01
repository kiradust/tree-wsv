# Tree Wasserstein singular vectors

This is repository for the ICLR 2025 paper [Fast unsupervised ground metric learning with tree-Wasserstein distance](https://openreview.net/forum?id=FBhKUXK7od). We embed data matrices on trees and leverage the tree-Wasserstein distance to efficiently learn ground metrics for both samples and features!

![image](https://github.com/user-attachments/assets/d79b3887-c653-4fa5-b2ce-5a6a3029c3d4)

## Related and modified repositories
1. We have modified parts of the [``treeOT`` repository](https://github.com/oist/treeOT?tab=readme-ov-file) from [Approximating 1-Wasserstein Distance with Trees](https://openreview.net/forum?id=Ig82l87ZVU), in particular the ClusterTree algorithm (to allow initialisation based on a custom input distance metric).
2. Interested users are also encouraged to review the [``wsingular`` repository](https://github.com/CSDUlm/wsingular) from [Unsupervised Ground Metric Learning Using Wasserstein Singular Vectors](https://proceedings.mlr.press/v162/huizing22a/huizing22a.pdf). We include and attribute parts of this code in ``tree-wsv.`` In particular, we compare our results to the standard Wasserstein Singular Vector and Sinkhorn Singular Vector algorithms implemented by Huizing et al. on a genomics PBMC dataset that was shared by the authors.

## Set-up
Once you have cloned the repository, set up a virtual environment using the listed requirements.
```
sudo pip install -r requirements.txt
```
Illustrative vignettes will be added to this repository as it is updated and commented.

## Citation (note: not yet available)
```
@inproceedings{
dusterwald2025,
title     = {Fast unsupervised ground metric learning with tree-Wasserstein distance},
author    = {Kira M. D\"usterwald, Samo Hromadka and Makoto Yamada},
booktitle = {The Thirteenth International Conference on Learning Representations, {ICLR} 2025, Singapore},
publisher = {OpenReview.net},
year      = {2025},
url       = {https://openreview.net/forum?id=FBhKUXK7od}
}
```

## Datasets
The PBMC-3k preprocessed dataset was kindly shared by Huizing et al., 2002, and can be found on [figshare](https://figshare.com/s/b4904dfc0898e3837c77). Other preprocessed datasets are available on request.

## Contact
E-mail: kira (dot) dusterwald (dot) 21 (at) ucl.ac.uk
