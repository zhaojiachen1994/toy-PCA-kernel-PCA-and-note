import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_classification
from sklearn import decomposition

def PCA(X,d):
    X_mat = np.mat(X - X.mean(axis=0))
    Y_mat = np.mat(Y - Y.mean(axis=0))
    C_X = np.dot(X_mat.T, X_mat)/(X_mat.shape[0])
    C_Y = np.dot(Y_mat.T, Y_mat)/(X_mat.shape[0])
    M = np.dot(C_Y.I, C_X)
    U, sigma, VT = np.linalg.svd(M)
    U = U[:,0:d].T
    X_new = np.dot(U, X.T)
    return X_new

#-------Step1.make the classification--------#
X,Y = make_classification(n_samples=200,n_features=3,n_redundant=1,n_repeated=0,n_informative=2,n_clusters_per_class=1,n_classes=2)
fig1 = plt.figure()
ax = Axes3D(fig1)
ax.set_title('Samples in the original space')
ax.scatter(X[:,0],X[:,1],X[:,2],marker='o',c=Y)


#-------Step2. projection to 2-dim space by PCA in sklearn----------
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_1 = pca.transform(X)
fig2 = plt.figure()
plt.clf()
plt.title('Samples in feature space decomposed by sklearn')
plt.scatter(X_1[:,0],X_1[:,1],c=Y)
#------Step3. PCA to 2-dim space by my-PCA------------#

d=2                                     # set the target number of PC
X=X-np.mean(X,axis=0)                   # centralize the samples
C=np.mat(np.dot(X.T,X)/X.shape[0])      # compute the covariance matrix
U, sigma, VT = np.linalg.svd(C)         # decompose C by SVD
PC = VT[0:d,:]                          # pick the principal components
X_2 = np.dot(PC,X.T).T                  # samples in a lower-dimension space

fig3 = plt.figure()
plt.clf()
plt.title('Samples in feature space decomposed by my PCA')
plt.scatter(X_2[:,0],X_2[:,1],c=Y)

plt.show()