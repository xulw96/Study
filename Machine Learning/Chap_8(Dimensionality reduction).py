import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# projection datasets
np.random.seed(618)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
# PCA with SVD
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)  # center before SVD
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
m, n = X.shape
S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)
np.allclose(X_centered, U.dot(S).dot(Vt))
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
X2D_using_svd = X2D
# PCA with Scikit-learn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)  # auto-center the data
X3D_inv = pca.inverse_transform(X2D)  # reverse back to original matrix; with data loss
np.mean(np.sum(np.square(X3D_inv - X), axis=1))
X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])
principal_component = pca.components_
explained_variance = pca.explained_variance_ratio_  # explained variance
# 3d drawing
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]
x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)
C = pca.components_
R = C.T.dot(C)
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])
from mpl_toolkits.mplot3d import Axes3D
def plot_3d():
    fig = plt.figure(figsize=(6, 3.8))
    ax = fig.add_subplot(111, projection='3d')
    X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
    X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]
    ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], 'bo', alpha=0.5)
    ax.plot_surface(x1, x2, z, alpha=0.2, color='k')
    np.linalg.norm(C, axis=0)
    ax.add_artist(Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle='-|>', color='k'))
    ax.add_artist(Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle='-|>', color='k'))
    ax.plot([0], [0], [0], 'k.')
    for i in range(m):
        if X[i, 2] > X3D_inv[i, 2]:
            ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], 'k-')
        else:
            ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], 'k-', color='#505050')
    ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], 'k+')
    ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], 'k.')
    ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], 'bo')
    ax.set_xlabel('$x_1$', fontsize=18)
    ax.set_ylabel('$x_2$', fontsize=18)
    ax.set_zlabel('$x_3$', fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])
    plt.show()
def plot_2d():
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(X2D[:, 0], X2D[:, 1], 'k+')
    ax.plot(X2D[:, 0], X2D[:, 1], 'k.')
    ax.plot([0], [0], 'ko')
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$z_1$', fontsize=18)
    ax.set_ylabel('$z_2$', fontsize=18, rotation=0)
    ax.axis([-1.5, 1.3, -1.2, 1.2])
    ax.grid(True)
    plt.show()
# Manifold learning
from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=618)
def plot_swiss_roll():
    axes = [-11.5, 14, -2, 23, -12, 15]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel('$x_1$', fontsize=18)
    ax.set_ylabel('$x_2$', fontsize=18)
    ax.set_zlabel('$x_3$', fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])
    plt.title('swiss_roll_plot', fontsize=18)
    plt.show()
def plot_squished_swiss_roll():
    axes = [-11.5, 14, -2, 23, -12, 15]
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
    plt.axis(axes[:4])
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18, rotation=0)
    plt.grid(True)

    plt.subplot(122)
    plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel('$z_1', fontsize=18)
    plt.grid(True)
    plt.show()
def plot_manifold_decision_boundary():
    axes = [-11.5, 14, -2, 23, -12, 15]
    x2s = np.linspace(axes[2], axes[3], 10)
    x3s = np.linspace(axes[4], axes[5], 10)
    x2, x3 = np.meshgrid(x2s, x3s)
    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection='3d')
    positive_class = X[:, 0] > 5
    X_pos = X[positive_class]
    X_neg = X[~positive_class]
    ax.view_init(10, -70)
    ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
    ax.plot_wireframe(5, x2, x3, alpha=0.5)
    ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])
    plt.show()
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    plt.plot(t[positive_class], X[positive_class, 1], "gs")
    plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
    plt.show()
    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection='3d')
    positive_class = 2 * (t[:] - 4) > X[:, 1]
    X_pos = X[positive_class]
    X_neg = X[~positive_class]
    ax.view_init(10, -70)
    ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
    ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])
    plt.show()
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    plt.plot(t[positive_class], X[positive_class, 1], "gs")
    plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
    plt.plot([4, 15], [0, 22], "b-", linewidth=2)
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
    plt.show()
# PCA
angle = np.pi / 5
stretch = 5
m = 200
np.random.seed(3)
X = np.random.randn(m, 2) / 10
X = X.dot(np.array([[stretch, 0], [0, 1]]))  # stretch the matrix
X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])  # rotate
u1 = np.array([np.cos(angle), np.sin(angle)])
u2 = np.array([np.cos(angle - 2 * np.pi / 6), np.sin(angle - 2 * np.pi / 6)])
u3 = np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])
X_proj1 = X.dot(u1.reshape(-1, 1))
X_proj2 = X.dot(u2.reshape(-1, 1))
X_proj3 = X.dot(u3.reshape(-1, 1))
def plot_pca_projection():
    plt.figure(figsize=(8, 4))
    plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    plt.plot([-1.4, 1.4], [1.4 * u1[1] / u1[0], 1.4 * u1[1] / u1[0]], 'k-', linewidth=1)
    plt.plot([-1.4, 1.4], [-1.4 * u2[1] / u2[0], 1.4 * u2[1] / u2[0]], 'k--', linewidth=1)
    plt.plot([-1.4, 1.4], [1.4 * u3[1] / u3[0], 1.4 * u3[1] / u3[0]], 'k:', linewidth=2)
    plt.plot(X[:, 0], X[:, 1], 'bo', alpha=0.5)
    plt.plot([-1.4, 1.4, -1.4, 1.4])
    plt.axis([-1.4, 1.4, -1.4, 1.4])
    plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    plt.text(u1[0] + 0.1, u1[1] - 0.05, r'$\mathbf{c_1}$', fontsize=22)
    plt.text(u3[0] + 0.1, u3[1], r'$\mathbf{c_2}$', fontsize=22)
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18, rotation=0)
    plt.grid(True)

    plt.subplot2grid((3, 2), (0, 1))
    plt.plot([-2, 2], [0, 0], 'k-', linewidth=1)
    plt.plot(X_proj1[:, 0], np.zeros(m), 'bo', alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])
    plt.grid(True)

    plt.subplot2grid((3, 2), (1, 1))
    plt.plot([-2, 2], [0, 0], 'k--', linewidth=1)
    plt.plot(X_proj2[:, 0], np.zeros(m), 'bo', alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])
    plt.grid(True)

    plt.subplot2grid((3, 2), (2, 1))
    plt.plot([-2, 2], [0, 0], 'k:', linewidth=2)
    plt.plot(X_proj3[:, 0], np.zeros(m), 'bo', alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.axis([-2, 2, -1, 1])
    plt.xlabel('$z_1$', fontsize=18)
    plt.grid(True)
    plt.title('pca_best_projection')
    plt.show()
# MNIST compression
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
from sklearn.model_selection import train_test_split
X = mnist['data']
y = mnist['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmin(cumsum >= 0.95) + 1  # least vectors to achieve 95% variance.
pca = PCA(n_components=0.95)  # do the selection as above
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)  # recover the matrix
def plot_digit(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis('off')
def plot_compression():
    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plot_digit(X_train[::2100])
    plt.title('Original', fontsize=16)
    plt.subplot(122)
    plot_digit(X_recovered[::2100])
    plt.title('Compressed', fontsize=16)
    plt.show()
# Incremental PCA; for large datasets
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print('.', end='')
    inc_pca.partial_fit(X_batch)  # call partial fit for batch-fitting; similar to warmstart
X_reduced = inc_pca.transform(X_train)
X_recovered = inc_pca.inverse_transform(X_reduced)
def plot_incremental_pca():
    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plot_digit(X_train[::2100])
    plt.subplot(122)
    plot_digit(X_recovered[::2100])
    plt.tight_layout()
    plt.show()
# achieve incremental by memmap()
filename = 'my_mnist.data'
m, n = X_train.shape
X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)  # able to call just fit
del X_mm
# randomdized PCA
rnd_pca = PCA(n_components=154, svd_solver='randomized', random_state=618)
X_reduced = rnd_pca.fit_transform(X_train)
# different n_compoents
import time
for n_components in (2, 10, 154):
    print('n_components =', n_components)
    regular_pca = PCA(n_components=n_components)
    inc_pca = IncrementalPCA(n_components=n_components, batch_size=500)
    rnd_pca = PCA(n_components=n_components, random_state=618, svd_solver='randomized')
    for pca in (regular_pca, inc_pca, rnd_pca):
        t1 = time.time()
        pca.fit(X_train)
        t2 = time.time()
        print('{}: {:.1f} seconds'.format(pca.__class__.__name__, t2-t1))
# different dataset's size
times_rpca = []
times_pca = []
sizes = [1000, 10000, 20000, 30000, 40000, 50000, 70000, 100000, 200000, 500000]
for n_samples in sizes:
    X = np.random.randn(n_samples, 5)
    pca = PCA(n_components=2, svd_solver='randomized', random_state=618)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_rpca.append(t2 - t1)
    pca = PCA(n_components=2)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_pca.append(t2 - t1)
def plot_comparison_graph():
    fig = plt.figure()
    plt.plot(sizes, times_rpca, 'b-o', label='RPCA')
    plt.plot(sizes, times_pca, 'r-s', label='PCA')
    plt.xlabel('n_samples')
    plt.ylabel('Training time')
    plt.legend(loc='upper left')
    plt.title('PCA and Randomized PCA time complexity')
    plt.show()
# different features
times_rpca = []
times_pca = []
sizes = [1000, 2000, 3000, 4000, 5000, 6000]
for n_features in sizes:
    X = np.random.randn(2000, n_features)
    pca = PCA(n_components=2, random_state=618, svd_solver='randomized')
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_rpca.append(t2 - t1)
    pca = PCA(n_components=2)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_pca.append(t2 - t1)
def plot_feature_graph():
    plt.plot(sizes, times_rpca, 'b-o', label='RPCA')
    plt.plot(sizes, times_pca, 'r-s', label='PCA')
    plt.xlabel('n_features')
    plt.ylabel('Training time')
    plt.legend(loc='upper left')
    plt.title('PCA and Randomized PCA time complexity')
    plt.show()
# Kernel PCA
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=618)
from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
lin_pca = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True)  # build model to learn reconstruction; thus enable inverse_transform
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.001, coef0=1, fit_inverse_transform=True)
def plot_KPCA():
    y = t > 6.9
    plt.figure(figsize=(11, 4))
    for subplot, pca, title in ((131, lin_pca, 'linear kernel'), (132, rbf_pca, 'RBF kernel, $\gamma=0.004$'),
                                (133, sig_pca, 'Sigmoid kernel, $\gamma=10^{-3}, r=1$')):
        X_reduced = pca.fit_transform(X)
        if subplot == 132:
            X_reduced_rbf = X_reduced
        plt.subplot(subplot)
        plt.plot(X_reduced[y, 0], X_reduced[y, 1], 'gs')
        plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], 'y^')
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel('$z_1$', fontsize=18)
        if subplot == 131:
            plt.ylabel('$z_2$', fontsize=18, rotation=0)
        plt.grid(True)
    plt.show()
# minimize pre-image reconstruction error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
clf = Pipeline([('kpca', KernelPCA(n_components=2)), ('log_reg', LogisticRegression(solver='liblinear'))])
param_grid = [{'kpca_gamma': np.linspace(0.03, 0.05, 10), 'kpca_kernel': ['rbf', 'sigmoid']}]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print(grid_search.best_params_)
# LLE
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=618)
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=618)
def plot_lle_unrolling():
    X_reduced = lle.fit_transform(X)
    plt.title('Unrolled swiss roll using LLE', fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel('$z_1$', fontsize=18)
    plt.ylabel('$z_2$', fontsize=18)
    plt.axis([-0.065, 0.055, -0.1, 0.12])
    plt.grid(True)
    plt.show()
# MDS, Isomap and t_SNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
mds = MDS(n_components=2, random_state=618)
isomap = Isomap(n_components=2)
tsne = TSNE(n_components=2, random_state=618)
X_reduced_mds = mds.fit_transform(X)
X_reduced_isomap = isomap.fit_transform(X)
X_reduced_tsne = tsne.fit_transform(X)
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_mnist = mnist['data']
y_mnist = mnist['target']
lda.fit(X_mnist, y_mnist)
X_reduced_lda = lda.transform(X_mnist)
def plot_manifolds_graph():
    titles = ['MDS', 'Isomap', 't-SNE']
    plt.figure(figsize=(11, 4))
    for subplot, title, X_reduced in zip((131, 132, 133), titles,
                                         (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
        plt.subplot(subplot)
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel('$z_1$', fontsize=18)
        if subplot == 131:
            plt.ylabel('$z_2$', fontsize=18, rotation=0)
        plt.grid(True)
    plt.show()
# Clustering/ GaussianMixture
from sklearn.mixture import GaussianMixture
y_pred = GaussianMixture(n_components=3, random_state=618).fit(X).predict(X)
mapping = np.array([2, 0, 1])
y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])
def plot_GaussianMixture():
    plt.plot(X[y_pred==0, 2], X[y_pred==0, 3], 'yo', label='Cluster 1')
    plt.plot(X[y_pred==1, 2], X[y_pred==1, 3], 'bs', label='Cluster 2')
    plt.plot(X[y_pred==2, 2], X[y_pred==2, 3], 'g^', label='Cluster 3')
    plt.xlabel('Petal length', fontsize=14)
    plt.ylabel('Petal width', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.show()
print(np.sum(y_pred==y) / len(y_pred))
# K-Means
from sklearn.datasets import make_blobs
blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8],
                         [-2.8, 1.8], [-2.8, 1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=618)
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14, rotation=0)
    plt.show()

from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=618)
y_pred = kmeans.fit_predict(X)
print(kmeans.cluster_centers_, kmeans.labels_)
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)




