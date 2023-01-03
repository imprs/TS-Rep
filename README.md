
# TS-Rep: Self-supervised time series representation learning from robot sensor data

This is the code for the paper [TS-Rep: Self-supervised time series representation learning from robot sensor data](https://sslneurips22.github.io/paper_pdfs/paper_74.pdf) presented at the [Self-Supervised Learning - Theory and Practice](https://sslneurips22.github.io/) workshop at [NeurIPS](https://neurips.cc/) 2022.

**Acknowledgement:** This code is developed on top of the T-Loss code [github](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries) ([paper](https://papers.nips.cc/paper/2019/hash/53c6de78244e9f528eb3e1cda69699bb-Abstract.html)). We thank the authors of T-Loss for making their code available.


## Requirements
Install dependencies using the Conda environment:
```
conda env create -f environment.yml
conda activate ts-rep
```


## Datasets
All pre-processed datasets in `.npz` can be downloaded with a single command:

`bash datasets/download_datasets.sh`

*Note:* Pre-processing scripts for the raw datasets will be uploaded soon.


## Usage

Scripts are provided in `scripts/` to run the experiments.

For example, to train `TS-Rep` on the Manipulation dataset run,

`bash scripts/manipulation.sh`


### Dataset type
TS-Rep supports both fixed-length (i.e., padded) and varying length time series. In each script, the fixed-length dataset is the default. To run on a varying-length/ NaN padded dataset, we have to uncomment a few lines. Again, taking the Manipulation dataset as an example, uncomment the following lines in `scripts/manipulation.sh`:

```
# dataset_name="manipulation_std_nan_padded"
# dataset_dir="datasets/manipulation/varying"
```

### Tasks
Arguments to run these tasks are provided in the individual scripts; for example in `manipulation.sh`, we have clustering and anomaly detection tasks.

- Clustering
- Anomaly Detection
- Terrain Classification

### Output
Output will be stored in `output/%m_%d_%H_%M_%S/` directories, and based on the task, `CSV` and `PNG/ EPS` files will be generated.