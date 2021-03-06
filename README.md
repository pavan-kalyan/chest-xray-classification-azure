# Chest X-ray classification using Azure services

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](code-of-conduct.md)
## Project Objective
The purpose of this project is to design, train and deploy a deep learning model on Azure Machine learning service which can diagnose a chest X-ray scan.
The results of the diagnosis from the model can potentially serve as a preliminary diagnosis in a medical facility, resulting in reduced workload for radiologists and money and time saved for the medical facility.

A convenient webapp has been developed for easy and quick diagnosis of any chest X-ray scan. The source code for this is available [here](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/chexray-web-app).

## Project Description
This project uses the NIH dataset for chest X-ray scans available [here](https://nihcc.app.box.com/v/ChestXray-NIHCC).

This project uses the NASnet architecture trained on ImageNet as a weights initializer, with a few layers added to adapt to the chest X-ray Dataset.
The model have been trained using pretrained weights for transfer learning and also as a weights initializer.
This model achieved 0.8182 AUC at 50 epochs compared to the original stanford paper which achieved 0.841 AUC.


## Results

|Model | AUC | Epochs |
|------|:---:|:------:|
|NASNet|0.8182|50|

#### Per Disease AUC Score

| Disease      | AUC Score | Disease            | AUC Score |
|--------------|-----------|--------------------|-----------|
| Atelectasis  | 0.796015  | Pneumothorax       | 0.866539 |
| Cardiomegaly | 0.866409  | Consolidation      | 0.77846 |
| Effusion     | 0.857553 | Edema              | 0.837394 |
| Infiltration | 0.717564  | Emphysema          | 0.915356  |
| Mass         | 0.841312  | Fibrosis           | 0.812732  |
| Nodule       | 0.767547 | Pleural Thickening | 0.776329  |
| Pneumonia    | 0.731321  | Hernia             | 0.877378 |


## Quick Start
If you just want to get the data, process it and train the model then run the scripts in the src/ folder in the following order after cloning this repo.
1. get_and_extract_data.py
2. data_prep.py
3. pytorch_train.py
The necessary packages are present in the env.yml file. It is recommended to use anaconda to setup the environment for you, using the env.yml file.

pytorch_run.py is a scoring script used with Azure machine learning service for the deployed model and the chest X-ray classfication web app.

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. To utilize Azure Machine Learning Service run the [0_setup_aml notebook](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/notebook/0_setup_aml.ipynb) which will use your azure credentials and setup the Azure Machine learning workspace and storage account.
2. Raw Data will be kept [here](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/data/raw) within this repo. It can be downloaded by running [src/get_and_extract.py](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/src/get_and_extract_data.py)
or by running notebook/[1_get_and_extract notebook](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/src/get_and_extract_data.ipynb)
3. For data processing and preparation, run [2_data_prep notebook](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/notebook/2_data_prep.ipynb) or [src/data_prep.py](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/src/get_and_extract_data.py). This will process the dataset and store the train/validate/test split in data/processed/.
4. (a) For training the model on Azure's compute target, run the [3_pytorch_train notebook](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/notebook/3_pytorch_train.ipynb) notebook, this will setup the experiment and automatically procure a compute target and train the model there and store the output on the Azure workspace.
4. (b) For training the model on your local system or a VM (e.g [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-in/services/virtual-machines/data-science-virtual-machines/)), simply run the following
    ```
    python src/pytorch_train.py --data-dir="data/processed/images" 
    ```
5. For Deploying the model on Azure, run the [4_pytorch_run noteboook](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/notebook/4_pytorch_run.ipynb). This will register the model,create the image and deploy on Azure Container Instances. The link to the deployment will be available on the azure portal.

## Disclaimer
This x-ray image pathology classification system is neither intended for real clinical diagnosis nor to be used as a substitute for a medical professional. The performance of this model for clinical use has not been established.


## Acknowledgements
With deep gratitude to researchers and developers at PyTorch, NIH, Stanford, Project Jupyter and [AzureChestXray repo](https://github.com/Azure/AzureChestXRay). With special thanks to my mentors at Microsoft for their guidance.

## Contributing
This project welcomes contributions and suggestions. Please put up a Github issue before working on a bug or a feature, to avoid redundant work.

This project has adopted the Contributor Covenant Code of Conduct. For more information see [Code of Conduct FAQ](https://www.contributor-covenant.org/faq)
