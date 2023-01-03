import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score

def cluster(seq, true_num_clusters, algorithm):
    if algorithm=="DBSCAN":
        pred_labels = DBSCAN(eps=0.05, min_samples=2, metric='cosine').fit_predict(seq)
    if algorithm=="KMEANS":
        pred_labels = KMeans(n_clusters=5).fit_predict(seq)
    elif algorithm=="DPGMM":
        bgm = mixture.BayesianGaussianMixture(n_components=15, covariance_type='full', weight_concentration_prior=0.01, max_iter=300,n_init=10,tol=1e-3, verbose=0).fit(seq)
        pred_labels = bgm.predict(seq)
        #print(np.around(bgm.weights_,decimals=4))
    elif algorithm=="GMM":
        gmm = mixture.GaussianMixture(n_components=true_num_clusters, covariance_type='full', max_iter=300,n_init=10,tol=1e-3, verbose=0).fit(seq)
        pred_labels = gmm.predict(seq)
    n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    print("number of clusters is: ", n_clusters)
    # print(np.unique(pred_labels))
    return pred_labels, n_clusters

def get_pca_tsne(X_latent):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_latent)
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000, n_jobs=12)
    X_tsne = tsne.fit_transform(X_latent)

    return X_pca, X_tsne

def perform_clustering(X_latent, Y_true, true_num_clusters):
    clustering_pred = {"GMM": []}#, "DPGMM": []}
    clustering_metrics = {"GMM": {}}#, "DPGMM": {}}
    for algorithm in list(clustering_pred.keys()):
        #algorithm = 'DPGMM'
        Y_pred, n_cluster_det = cluster(X_latent, true_num_clusters, algorithm=algorithm)
        NMI = metrics.normalized_mutual_info_score(Y_true, Y_pred, average_method='arithmetic') # to match the version of v0.22 in other methods
        AMI = metrics.adjusted_mutual_info_score(Y_true, Y_pred)
        OLD_NMI = metrics.normalized_mutual_info_score(Y_true, Y_pred)        
        s_score = silhouette_score(X_latent, Y_pred)
        DBI = davies_bouldin_score(X_latent, Y_pred)
        # print(f'old_nmi: {OLD_NMI}')
        print(f"n_cluster_det: {n_cluster_det}, NMI: {NMI}, AMI: {AMI}, s_score: {s_score}, DBI: {DBI}")

        clustering_dict = {"NMI": NMI, "AMI": AMI, "n_clusters": n_cluster_det, "s_score": s_score, "DBI": DBI}

        clustering_pred[algorithm].extend(Y_pred.tolist())

        clustering_metrics[algorithm].update(clustering_dict)

    return clustering_pred,  clustering_metrics

def plot_clustersing(x, y_true, clustering_preds, clustering_metrics, epoch, start_time):
    """
    Args:
        X: T-SNE output
        y_true: true cluster labels
        clustering_preds: Dict of algorithms with list of cluster predicted assignments
        clustering_metrics: Dict of algorithms containing dict of metrics
        epoch: epoch number
        start_time: for incremental dataset
    """
    algorithms = list(clustering_preds.keys())
    n_algo = len(algorithms)
    fig, ax = plt.subplots(1, n_algo+1, figsize=(8 * (n_algo+1), 8), dpi=300)
    cmap = cm.rainbow(np.linspace(0, 1, len(np.unique(y_true))))

    #ax[0].set_title(f'Epoch {epoch}: T-SNE | True labels | time slice: {start_time}') #TODO: temp removed for paper
    ax[0].set_xlabel("tsne feature 1")
    ax[0].set_ylabel("tsne feature 2")
                
    for cid in np.unique(y_true):
        cidx = np.where(y_true==cid)[0]
        cols = cmap[np.tile(cid,len(cidx))]
        ax[0].scatter(x[cidx,0], x[cidx,1],c=cols,label=cid)
    ax[0].legend(bbox_to_anchor=(1.14,1))


    for i, algo in enumerate(algorithms):
        y_pred = clustering_preds[algo]
        cmap_pred = cm.rainbow(np.linspace(0, 1, len(np.unique(y_pred))))
        clustering_dict = clustering_metrics[algo]
        NMI, DBI = np.round(clustering_dict["NMI"],4), np.round(clustering_dict["DBI"],4)
        s_score = np.round(clustering_dict["s_score"],4)

        ax[i+1].set_title(f'Epoch {epoch}: Pred labels {algo} | NMI: {NMI} | s_score: {s_score:.4f} | DBI: {DBI}')
        ax[i+1].set_xlabel("tsne feature 1")
        ax[i+1].set_ylabel("tsne feature 2")

        for j,cid in enumerate(np.unique(y_pred)):
            cidx = np.where(y_pred==cid)[0]
            cols = cmap_pred[np.tile(j,len(cidx))] #dpgmm doesn't alway assign cluster numbers from 1,2,3,..,n, it drops clusters in between.
            # print(cid, len(cidx))
            ax[i+1].scatter(x[cidx,0], x[cidx,1],c=cols,label=cid)
        ax[i+1].legend(bbox_to_anchor=(1.14,1))

    return plt, fig

def plot_clustersing_paper(x, y_true, plot_axis_label):
    """
    Args:
        X: T-SNE output
        y_true: true cluster labels
        clustering_preds: Dict of algorithms with list of cluster predicted assignments
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8),  dpi=300)
    cmap = cm.rainbow(np.linspace(0, 1, len(np.unique(y_true))))

    #ax[0].set_title(f'Epoch {epoch}: T-SNE | True labels | time slice: {start_time}') #TODO: temp removed for paper
    if plot_axis_label:
        ax.set_xlabel("TSNE feature 1", fontsize=31, labelpad=30)
    ax.set_ylabel("TSNE feature 2", fontsize=31, labelpad=30)
                
    for cid in np.unique(y_true):
        cidx = np.where(y_true==cid)[0]
        cols = cmap[np.tile(cid,len(cidx))]
        ax.scatter(x[cidx,0], x[cidx,1],c=cols,label=cid)
    #ax.legend(bbox_to_anchor=(1.14,1))
    ax.tick_params(labelsize=30, pad=8)
    fig.tight_layout()

    return plt, fig
