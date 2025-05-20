# DSSG-DTI
DSSG-DTI: Dual-Stream Synergistic Attention with Sparsity-Guided Feature Fusion for Drug–Target Interaction Prediction.

# Cite
## Setup
The project setup includes installing dependencies, preparing the dataset and python version 3.8.

### Installing dependencies
dgl==1.0.2

dgllife==0.3.2

einops==0.8.0

numpy==1.24.4

pandas==2.0.3

rdkit==2024.3.5

scikit-learn==1.3.2

scipy==1.10.1

torch==2.2.1

torchaudio==2.2.1

torchvision==0.17.1

tqdm==4.67.1

yacs==0.1.8

zipp==3.20.2

### Preparing Dataset
The datasets folder contains all experimental data utilized in DSSG-DTI, including the BindingDB [1], BioSNAP [2], and Human [3] datasets. These datasets are provided in CSV format and consist of three columns: **SMILES**, **Protein** and **Y**.

## Run
### For Human dataset
`$ python main.py --cfg ./configs/human.yaml --outname human_model --data human --num_worker 0`
### For BioSNAP dataset
`$ python main.py --cfg ./configs/biosnap.yaml --outname biosnap_model --data biosnap --num_worker 0`
### For BindingDB dataset
`$ python main.py --cfg ./configs/bindingdb.yaml --outname bindingdb_model --data bindingdb --num_worker 0`

## Reference
>[1] Bai, P.; Miljković, F.; Ge, Y.; Greene, N.; John, B.; Lu, H. Hierarchical Clustering Split for Low-Bias Evaluation of Drug-Target Interaction Prediction. 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). 2021; pp 641–644.

>[2] Zitnik, M.; Sosic, R.; Leskovec, J. BioSNAP Datasets: Stanford Biomedical Network Dataset Collection. http://snap.stanford.edu/biodata 2018, 5.

>[3] Liu, H.; Sun, J.; Guan, J.; Zheng, J.; Zhou, S. Improving Compound–Protein Interaction Prediction by Building up Highly Credible Negative Samples. Bioinformatics 2015, 31, i221–i229.
