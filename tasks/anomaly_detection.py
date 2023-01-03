import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
# from sklearn.metrics import ConfusionMatrixDisplay


def in_range(mean, std, test_array):
    higher_limit_test = (test_array >= (mean + 3 * std)) # if outside this limit then 1/anomalous
    lower_limit_test = (test_array <= (mean - 3 * std))
    return np.logical_or(higher_limit_test, lower_limit_test).astype(np.int)

def get_confusion_matrix(true_labels, pred_labels):
    conf_mat = metrics.confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = conf_mat.ravel()
    conf_mat_dict = {"tn" : tn, "fp": fp, "fn": fn, "tp": tp}
    print(conf_mat_dict)
    return conf_mat, conf_mat_dict

def get_prec_rec_f1(true_labels, pred_labels):
    prec = metrics.precision_score(true_labels, pred_labels)
    recall = metrics.recall_score(true_labels, pred_labels)
    f1_score_ = metrics.f1_score(true_labels, pred_labels)
    prec_rec_f1_dict = {"precision": prec, "recall": recall, "f1_score": f1_score_}
    print(prec_rec_f1_dict)
    return prec_rec_f1_dict

def get_auc(x,y):
    return metrics.auc(x, y)

def get_roc(true_labels, pred_scores):
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_scores, pos_label=1, drop_intermediate=False) #pos_label=anomaly=1 in our case
    return tpr, fpr, thresholds

def get_auroc(true_labels, pred_scores):
    auroc = metrics.roc_auc_score(true_labels, pred_scores)
    return auroc

def get_prc(true_labels, pred_scores):
    prc  = metrics.precision_recall_curve(true_labels, pred_scores, pos_label=1) #pos_label=anomaly=1 in our case
    prc_prec, prc_rec, prc_thres = prc
    prc_dict = {"prc_prec": prc_prec, "prc_rec": prc_rec, "prc_thres": prc_thres }
    return prc, prc_dict

def get_auprc(true_labels, pred_scores):
    prc,prc_dict = get_prc(true_labels, pred_scores)
    auprc = get_auc(prc_dict['prc_prec'], prc_dict['prc_rec'])
    return auprc

def get_avg_prec_score(true_labels, pred_scores):
    aupr = metrics.average_precision_score(true_labels, pred_scores)
    return aupr

# Taken from https://github.com/againerju/maad_highway/blob/6ba19548366c89fa7552b36b028fae710c04b8a9/src/evaluation.py#L200
def get_fpr_at_95tpr(true_labels, pred_scores):
    """ Compute Anomaly (AD) detction metric FPR-95%-TPR.
    """
    tpr, fpr,_ = get_roc(true_labels, pred_scores)

    hit = False
    tpr_95_lb = 0
    tpr_95_ub = 0
    fpr_95_lb = 0
    fpr_95_ub = 0

    for i in range(len(tpr)):
        if tpr[i] > 0.95 and not hit:
            tpr_95_lb = tpr[i - 1]
            tpr_95_ub = tpr[i]
            fpr_95_lb = fpr[i - 1]
            fpr_95_ub = fpr[i]
            hit = True

    s = pd.Series([fpr_95_lb, np.nan, fpr_95_ub], [tpr_95_lb, 0.95, tpr_95_ub])

    s = s.interpolate(method="index")

    return s.iloc[1]

def get_ad_metrics_paper(true_labels, pred_scores):
    auroc = get_auroc(true_labels, pred_scores)
    # auprc_abnormal = get_auprc(true_labels, pred_scores) # AUPR-abnormal
    # auprc_normal = get_auprc(1 - true_labels, -pred_scores) # AUPR-normal
    # Becuase in Paad and MAAD they do it like below!
    auprc_abnormal = get_avg_prec_score(true_labels, pred_scores) # AUPR-abnormal
    auprc_normal = get_avg_prec_score(1 - true_labels, -pred_scores) # AUPR-normal
    fpr_at_95tpr = get_fpr_at_95tpr(true_labels, pred_scores)
    ad_metrics_paper_dict = {"auroc":auroc ,"auprc_abnormal":auprc_abnormal , "auprc_normal": auprc_normal, "fpr_at_95tpr": fpr_at_95tpr}
    print(f'ad_metrics: {ad_metrics_paper_dict}')
    return ad_metrics_paper_dict


def anomaly_detection(model, train_data, train_labels, test_data, test_labels, dataset_name):
    if 'Boat' in dataset_name:
        nominal_class = 0 #TODO: parameterise
    elif 'Manipulation' in dataset_name:
        nominal_class = 2
    elif 'Qcat' in dataset_name:
        nominal_class = 0
    elif 'Putany' in dataset_name:
        nominal_class = 0
    else:
        raise ValueError(f'dataset_name={dataset_name} is not valid!')

    # extract nominal data from train set and fit OCSVM
    nominal_idx = np.where(train_labels==nominal_class)[0]
    if len(nominal_idx) == 0:
        print("len(nominal_idx) == 0")
        ValueError("No nominal data!! Failed! in len(nominal_idx) == 0")
    nominal_data = train_data[nominal_idx]
    nominal_labels = train_labels[nominal_idx]
    #get the representation of the nominal data
    train_nominal_features = model.encode(nominal_data, 100) #100 is batch size

    print(f'train_nominal_features shape: {train_nominal_features.shape}, nominal_labels shape: {nominal_labels.shape}')

    x_latent_train_norm = normalize(train_nominal_features) #normalize sample-wise using sklearn
    clf = OneClassSVM(kernel='rbf', gamma="scale").fit(x_latent_train_norm)
    scores_train = -clf.score_samples(x_latent_train_norm)
    scores_train_mean = np.mean(scores_train)
    scores_train_std = np.std(scores_train)
    print(f'mean: {scores_train_mean} std: {scores_train_std}')


    #get the representation of the test data and predict labels with OCSVM
    test_features = model.encode(test_data, 100)
    test_true_labels = test_labels.astype(np.int)
    # print(f'x_latent test shape: {test_features.shape}, labels test shape: {test_labels.shape}')

    x_latent_test_norm = normalize(test_features) #normalize sample-wise using sklearn
    scores = -clf.score_samples(x_latent_test_norm)
    pred = clf.predict(x_latent_test_norm)

    # convet true test labels to binary for evaluation.
    binary_test_labels = (test_labels != nominal_class).astype(np.int)

    # check if the score is with 3-std of the training scores.
    gauss_pred = in_range(scores_train_mean, scores_train_std, scores)

    # default pred labels from OCSVM, converted from -1 (outlier) and 1 (inlier) to 1 and 0, respectively.
    default_pred = (((-1 * pred) + 1)/2).astype(np.int)
    '''
    p_r_f1_default_dict = get_prec_rec_f1(binary_test_labels, default_pred)
    conf_mat_default_pred, conf_mat_default_dict = get_confusion_matrix(binary_test_labels, default_pred)

    p_r_f1_gauss_dict = get_prec_rec_f1(binary_test_labels, gauss_pred)
    conf_mat_gauss_pred, conf_mat_gauss_dict = get_confusion_matrix(binary_test_labels, gauss_pred)
    '''
    #pred_labels_dict = {"default_pred_labels": default_pred, "gauss_pred_labels": gauss_pred}
    pred_labels_dict = {"gauss_pred_labels": gauss_pred}

    ad_metrics_dict = {"test_labels": test_true_labels, "binary_test_labels": binary_test_labels, "raw_pred_score": scores, "default_pred": default_pred, "gauss_pred": gauss_pred}

    # Uncomment if want verbose results
    # for i in range(len(scores)):
    #     print(f'test_labels: {test_true_labels[i]}, binary_test_labels; {binary_test_labels[i]} raw_pred_score: {scores[i]}, default_pred: {default_pred[i]}, gauss_pred: {gauss_pred[i]}')

    return train_nominal_features, test_features,binary_test_labels, pred_labels_dict, ad_metrics_dict

def plot_anomaly_detection(x_train_tsne, x_test_tsne, y_true, pred_labels_dict):
    algorithms = list(pred_labels_dict.keys())
    n_algo = len(algorithms)
    fig, ax = plt.subplots(1, n_algo+1, figsize=(8 * (n_algo+1) , 8 ), dpi=300)
    cmap = cm.rainbow(np.linspace(0, 1, len(np.unique(y_true))))
    ax[0].set_title(f'T-SNE | True labels | Green (train nominals)')
    ax[0].set_xlabel("tsne feature 1", fontsize=19)
    ax[0].set_ylabel("tsne feature 2", fontsize=19)
     
    for cid in np.unique(y_true):
        cidx = np.where(y_true==cid)[0]
        cols = cmap[np.tile(cid,len(cidx))]
        ax[0].scatter(x_test_tsne[cidx,0], x_test_tsne[cidx,1],c=cols,label=cid)
    ax[0].scatter(x_train_tsne[:,0], x_train_tsne[:,1], c='green',label=2)
    ax[0].tick_params(labelsize=18)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5,1.12), fontsize=14, ncol=n_algo+1)


    for i, algo in enumerate(algorithms):
        y_pred = pred_labels_dict[algo]
        cmap_pred = cm.rainbow(np.linspace(0, 1, len(np.unique(y_pred))))

        ax[i+1].set_title(f'Pred labels {algo}')
        ax[i+1].set_xlabel("tsne feature 1", fontsize=19)
        # ax[i+1].set_ylabel("tsne feature 2", fontsize=19) because it's a 1X3 plot, we don't need label for Y axis!

        for j,cid in enumerate(np.unique(y_pred)):
            cidx = np.where(y_pred==cid)[0]
            cols = cmap_pred[np.tile(j,len(cidx))]
            # print(cid, len(cidx))
            ax[i+1].scatter(x_test_tsne[cidx,0], x_test_tsne[cidx,1],c=cols,label=cid)
        ax[i+1].scatter(x_train_tsne[:,0], x_train_tsne[:,1], c='green',label=2)
        ax[i+1].tick_params(labelsize=18)
        ax[i+1].legend(loc='upper center', bbox_to_anchor=(0.5,1.12), fontsize=14, ncol=n_algo+1)
    #fig.tight_layout()
    return plt, fig

def plot_confusion_matrix(conf_mat_dict):
    pred_methods = list(conf_mat_dict.keys())
    n_methods = len(pred_methods)
    fig, ax = plt.subplots(1, n_methods, figsize=(8 * (n_methods) , 8 ), dpi=300)
    
    for i, pred_method in enumerate(pred_methods):
        #plot confusion matrix
        conf_mat = conf_mat_dict[pred_method]
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        disp.plot(ax=ax[i], cmap="Blues", colorbar=False)
        ax[i].set_title("Confusion Matrix: {}".format(pred_method))

    return plt, fig