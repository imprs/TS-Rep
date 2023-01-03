# %% [markdown]
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# %% [markdown]
# # Qcat

# %%
import os
import math
import time

import scipy as sp
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import sklearn

import torch
import torch.nn as nn
import torch.optim as optim

import scikit_wrappers
from tasks.clustering import perform_clustering, get_pca_tsne, plot_clustersing, plot_clustersing_paper
from tasks.anomaly_detection import anomaly_detection, plot_anomaly_detection, plot_confusion_matrix, get_prec_rec_f1, get_confusion_matrix, get_ad_metrics_paper
from tasks.classification import classification, cross_validation
import datetime
import argparse


parser = argparse.ArgumentParser()

######################## Model parameters ########################
parser.add_argument('--dataset_dir', default='boat', type=str,
                    help='dataset path')
parser.add_argument('--dataset_name', default='boat', type=str,
                    help='Full name of the dataset')
parser.add_argument('--batch_size', default='64', type=int,
                    help='Full name of the dataset')
parser.add_argument('--modality', default='force_imu', type=str,
                    help='which modality to use force, imu or force_imu')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--gpu', default=0, type=int,
                    help='gpu number')
parser.add_argument('--out_channels', default=100, type=int,
                    help='out_channels for the network')
parser.add_argument('--clustering_task', action='store_true',
                    help='whether to run clustering task (default=False)')
parser.add_argument('--anomaly_detection_task', action='store_true',
                    help='whether to run anomaly_detection task (default=False)')
parser.add_argument('--classification_task', action='store_true',
                    help='whether to run classification task (default=False)')
parser.add_argument('--save_memory', action='store_true',
                    help='whether to run backward pass on postive representation \
                    first to save memory (default=False)')
args = parser.parse_args()

# %%
cuda = False
if torch.cuda.is_available():
    print("Using CUDA...")
    cuda = True

# arg reading
gpu = args.gpu # GPU number
out_channels = args.out_channels
batch_size = args.batch_size
modality = args.modality
clustering_task = args.clustering_task
anomaly_detection_task = args.anomaly_detection_task
classification_task = args.classification_task
save_memory = args.save_memory

# %%
output = './output/'
output_dir = os.path.join(output, datetime.datetime.now().strftime('%m_%d_%H_%M_%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# To store representations or models
models_dir = './models/'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# %% [markdown]
# ## Dataset

# %%
train_dataset = np.load(os.path.join(args.dataset_dir, f"{args.dataset_name}_training_set.npz"))
valid_dataset = np.load(os.path.join(args.dataset_dir, f"{args.dataset_name}_val_set.npz"))
test_dataset = np.load(os.path.join(args.dataset_dir, f"{args.dataset_name}_test_set.npz"))
val_test_dataset = np.load(os.path.join(args.dataset_dir, f"{args.dataset_name}_val_test_set.npz"))
# %%
train_data, train_labels = train_dataset['data'], train_dataset['labels']
valid_data, valid_labels = valid_dataset['data'], valid_dataset['labels']
test_data, test_labels = test_dataset['data'], test_dataset['labels']
val_test_data, val_test_labels = val_test_dataset['data'], val_test_dataset['labels']
print(f'train_data: {train_data.shape}, train_labels: {train_labels.shape}')
print(f'test_data: {test_data.shape}, test_labels: {test_labels.shape}')
print(f'val_test_data: {val_test_data.shape}, val_test_labels: {val_test_labels.shape}')

# convert from (batch, seq_len, features) to (batch, features, seq_len)
train_data = np.transpose(train_data, (0,2,1))
valid_data = np.transpose(valid_data, (0,2,1))
test_data = np.transpose(test_data, (0,2,1))
val_test_data = np.transpose(val_test_data, (0,2,1))

# %%
# Select modality
def filter_modality(modality_str, data):
    if modality_str == "force_imu":
        pass
    elif modality_str == "force":
        data = data[:,:12,:]
    elif modality_str == "imu":
        data = data[:,12:,:]
    else:
        raise ValueError(f'Invalid modality: {modality_str}')
    return data

def count_in_channels(modality_str):
    if modality_str == "force_imu":
        in_channels = 22
    elif modality_str == "force":
        in_channels = 12 # 0:12
    elif modality_str == "imu":
        in_channels = 10 # 12:22
    else:
        raise ValueError(f'Invalid modality: {modality_str}')
    return in_channels

train_data = filter_modality(modality, train_data)
valid_data = filter_modality(modality, valid_data)
test_data = filter_modality(modality, test_data)
val_test_data = filter_modality(modality, val_test_data)

in_channels = count_in_channels(modality)
print('\nApplying modality filter...\n')
print(f'train_data: {train_data.shape}, train_labels: {train_labels.shape}')
print(f'test_data: {test_data.shape}, test_labels: {test_labels.shape}')
print(f'val_test_data: {val_test_data.shape}, val_test_labels: {val_test_labels.shape}')

# %% [markdown]
# ### Learning Parameters

# %%
# Set to True to train a new model
training = True

# Prefix to path to the saved model
model = f'{output_dir}/Qcat'

# %%
hyperparameters = {
    "batch_size": batch_size,
    "channels": 40,
    "compared_length": None,
    "depth": 10,
    "nb_steps": 2000,
    "in_channels": in_channels,
    "kernel_size": 3,
    "penalty": None,
    "early_stopping": None,
    "lr": 0.001,
    "nb_random_samples": 10,
    "negative_penalty": 1,
    "out_channels": out_channels,
    "reduced_size": 80,
    "cuda": cuda,
    "gpu": gpu,
    "latent_shape": (len(train_data), out_channels)
}

# %%
true_num_clusters=6

# %% [markdown]
# ### Training

# %%
encoder = scikit_wrappers.CausalCNNEncoderClassifier()
encoder.set_params(**hyperparameters)

# %%
t = time.time()

if training:
    encoder.fit_encoder(train_data,save_memory=save_memory, verbose=True)
    encoder.save_encoder(model)
else:
    encoder.load_encoder(model)

# %%
torch.cuda.empty_cache()

# %% [markdown]
# ## Computing Representations
# We compute in the following (or load them from local storage if they are already precomputed) the learned representations.

# %%
compute_representations = True
train_encoded = 'models/Qcat_train_encoded.npy'

# %%
if compute_representations:
    train_features = encoder.encode(train_data, 100)
    np.save(train_encoded, train_features)
else:
    train_features = np.load(train_encoded)

t = time.time() - t
print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

# %%
epoch=50 # it runs for 50 epochs # TODO: it's just for print but fix this!

if clustering_task:
    print('\n Performing Clustering Task...\n')
    clustering_preds,  clustering_metrics = perform_clustering(train_features, train_labels, true_num_clusters)
    # get pca and tsne representations for plotting
    x_pca, x_tsne = get_pca_tsne(train_features)
    # get plt and fig
    plt, fig= plot_clustersing(x_tsne, train_labels, clustering_preds, clustering_metrics, epoch, 0)

    # log clustering epoch plot to neptune
    fig_spec = f'{output_dir}/epoch_plot_{epoch}'
    plt.savefig(f'{fig_spec}.eps', format='eps')
    print(f'{fig_spec}.eps')

    # #generate plot for the paper
    plt, fig = plot_clustersing_paper(x_tsne, train_labels, plot_axis_label=False) # X-axis label
    fig_spec = f'{output_dir}/true_label_plot_{epoch}'
    plt.savefig(f'{fig_spec}.eps', format='eps')

    # log NMI, AMI, SS, DBI
    clustering_metrics_df = pd.DataFrame([clustering_metrics['GMM']])
    file_path = f'{output_dir}/clustering_metrics.csv'
    clustering_metrics_df.to_csv (file_path, index = False, header=True)

    # log train feature representation for future
    output_rep_name = f'{output_dir}/feature_rep_train'
    np.savez(f'{output_rep_name}.npz', data = train_features, labels=train_labels, clustering_preds=clustering_preds["GMM"])

# %%
if anomaly_detection_task:
    print('\n Performing Anomaly Detection Task...\n')
    # perform anomaly detection
    x_latent_train,x_latent_test,true_labels, pred_labels_dict, ad_metrics_dict = anomaly_detection(\
        encoder, train_data, train_labels, val_test_data, val_test_labels, dataset_name="Qcat")

    # log anomaly detection plot
    _,x_train_tsne = get_pca_tsne(x_latent_train)
    _,x_test_tsne = get_pca_tsne(x_latent_test)
    plt, fig = plot_anomaly_detection(x_train_tsne, x_test_tsne, true_labels, pred_labels_dict)
    #save anomaly detetion figure
    fig_spec = f'{output_dir}/anomaly_detection'
    plt.savefig(f'{fig_spec}.png') #TODO: make it eps once results are ready!

    # log anomaly detection metrics
    conf_mats_dict = {}
    # check for both default labels and gauss labels
    for pred_key in list(pred_labels_dict.keys()):
        pred_labels = pred_labels_dict[pred_key]
        prec_rec_f1_dict = get_prec_rec_f1(true_labels, pred_labels)
        conf_mat_pred, conf_mat_dict = get_confusion_matrix(true_labels, pred_labels)
        conf_mats_dict.update({pred_key: conf_mat_pred}) # create a dict with pred_key and conf_mat_pred and append it

        # log prec, recall, f1
        prec_rec_f1_df = pd.DataFrame([prec_rec_f1_dict])
        file_path = f'{output_dir}/prec_rec_f1_ad.csv'
        prec_rec_f1_df.to_csv(file_path, index = False, header=True)

    # log ad metrics auroc, aupr, fpr@95tpr
    ad_metric_processed_dict = get_ad_metrics_paper(true_labels, ad_metrics_dict['raw_pred_score'])
    ad_metric_processed_df = pd.DataFrame([ad_metric_processed_dict])
    file_path = f'{output_dir}/ad_metrics.csv'
    ad_metric_processed_df.to_csv(file_path, index = False, header=True)

    # log val_test feature representations
    val_test_features = encoder.encode(val_test_data, 100)
    output_rep_name = f'{output_dir}/feature_rep_val_test'
    np.savez(f'{output_rep_name}.npz', data = val_test_features, labels=val_test_labels, raw_ad_pred=ad_metrics_dict['raw_pred_score'],
                                        default_pred=ad_metrics_dict['default_pred']) # gauss_pred can be calculated from raw score

# %%
# Classification
if classification_task:
    print('\n Performing Classification Task...\n')
    test_features = encoder.encode(test_data, 100)
    # For Clasification Test we evaluate on the holdout data so taking test data and labels.
    test_acc = classification(train_features, train_labels, test_features, test_labels)
    if not anomaly_detection_task:
        val_test_features = encoder.encode(val_test_data, 100)
    # For Cross-Validaion we need all the data so taking val_test data and labels.
    cv_acc = cross_validation(train_features, train_labels, val_test_features, val_test_labels)
    acc_dict = {"test_acc": test_acc, "cv_acc": cv_acc}

    # log cv_acc, test_acc
    classification_metrics_df = pd.DataFrame([acc_dict])
    file_path = f'{output_dir}/classification_metrics.csv'
    classification_metrics_df.to_csv (file_path, index = False, header=True)

    # log test feature representations
    output_rep_name = f'{output_dir}/feature_rep_test'
    np.savez(f'{output_rep_name}.npz', data = test_features, labels=test_labels)

# %%
# log model state
weights_path = f'{output_dir}/Qcat_CausalCNN_encoder'

if not clustering_task:
    # log train feature representation for future
    output_rep_name = f'{output_dir}/feature_rep_train'
    np.savez(f'{output_rep_name}.npz', data = train_features, labels=train_labels)

if not anomaly_detection_task:
    # log val_test feature representations
    val_test_features = encoder.encode(val_test_data, 100)
    output_rep_name = f'{output_dir}/feature_rep_val_test'
    np.savez(f'{output_rep_name}.npz', data = val_test_features, labels=val_test_labels)

if not classification_task:
    # log test feature representations
    test_features = encoder.encode(test_data, 100)
    output_rep_name = f'{output_dir}/feature_rep_test'
    np.savez(f'{output_rep_name}.npz', data = test_features, labels=test_labels)