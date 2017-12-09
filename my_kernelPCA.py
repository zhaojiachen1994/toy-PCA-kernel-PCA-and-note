# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import *

def rbf_kernel(X,Y=None,gamma=None):
    if gamma == None:
        gamma = 1.0/X.shape[1]
    K = euclidean_distances(X,Y,squared=True)
    K = -gamma*K
    np.exp(K,K)
    return K

def KernelCenterer(K):
    n_samples = K.shape[0]
    K_rows_mean = (np.sum(K, axis=0) / n_samples).reshape([1, -1])
    K_cols_mean = (np.sum(K, axis=1) / n_samples).reshape([-1, 1])
    K_all_mean = K_rows_mean.sum() / n_samples
    K_center = K - K_rows_mean - K_cols_mean + K_all_mean
    return K_center


#-------Step1.make the circles--------#
np.random.seed(0)
X,Y = make_circles(n_samples=100,noise=.05,factor=0.3)#样本数应该为偶数
fig = plt.figure(figsize=[12,12])
plt.subplot(2,2,1)
plt.title('Samples in the original space')
plt.scatter(X[:,0],X[:,1],c=Y)

#-------Step2. Projection to 2-dim space by PCA in sklearn----------
pca = PCA(n_components=2)
X_1 = pca.fit_transform(X)
plt.subplot(222)
plt.title('Projected by PCA in sklearn')
plt.scatter(X_1[:,0],X_1[:,1],c=Y)
x_new = np.array([1.0,0.0]).reshape([1,-1])

#-------Step3. Projection to 2-dim space by kernelPCA in sklearn----------
kpca = KernelPCA(kernel='rbf',fit_inverse_transform=True,gamma=3)
X_kpca = kpca.fit_transform(X)
plt.subplot(223)
plt.title('Projected by kernel PCA in sklearn')
plt.scatter(X_kpca[:,0],X_kpca[:,1],c=Y)

# -------Step4. Projection to 2-dim space by my kernelPCA----------
d = 2                               # the target dimension
K = rbf_kernel(X,gamma=3)           # compute the rbf kernel matrix
K_center = KernelCenterer(K)        # centralize the kernel matrix
D,U = np.linalg.eig(K_center)       # compute the eigenvalues and corresponding eigenvectors
indices = D.argsort()[::-1]         # sort eigenvectors in descending order
D = D[indices]                      # ...
U = U[:,indices]                    # ...
PC = U[:,:d]                        # sort eigenvectors in descending order
X_proj = np.dot(K_center,PC)        # project the samples in feature space


plt.subplot(224)
plt.title('Projected by my kernel PCA')
plt.scatter(X_proj[:,0],X_proj[:,1],c=Y)

plt.show()
