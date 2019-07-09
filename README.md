# Chest X-ray classification using Azure services


## Project Objective
The purpose of this project is to design, train and deploy a deep learning model on Azure Machine learning service which can diagnose a chest X-ray scan.
The results of the diagnosis from the model can potentially serve as a preliminary diagnosis in a medical facility, resulting in reduced workload for radiologists and money and time saved for the medical facility.

A convenient website has been developed for easy and quick diagnosis of any chest X-ray scan.

## Project Description
This project uses the NIH dataset for chest X-ray scans available [here](https://nihcc.app.box.com/v/ChestXray-NIHCC).

This project uses the NASnet architecture trained on ImageNet as a weights initializer, with a few layers added to adapt to the chest X-ray Dataset.
This model achieved 0.81 AUC at 15 epochs compared to the original stanford paper which achieved 0.84 AUC.


## Results

|Model | AUC | Epochs | Transfer Learning |
|------|:---:|:------:|-----------------:|
|NASNet|0.8117|20|No|
|NASNet|0.7064|30|Yes|

| Disease      | AUC Score | Disease            | AUC Score |
|--------------|-----------|--------------------|-----------|
| Atelectasis  | -  | Pneumothorax       | -  |
| Cardiomegaly | -  | Consolidation      | -  |
| Effusion     | - | Edema              | - |
| Infiltration | -  | Emphysema          | -  |
| Mass         | -  | Fibrosis           | -  |
| Nodule       | - | Pleural Thickening | -  |
| Pneumonia    | -  | Hernia             | -  |


## Quick Start
If you just want to get the data, process it, train the model then run the scripts in the src/ folder in the following order after cloning this repo.
1. get_and_extract_data.py
2. data_prep.py
3. pytorch_train.py
The necessary packages are present in the env.yml file. It is recommended to use anaconda to setup the environment for you.

pytorch_run.py is a scoring script used with Azure machine learning service for the deployed model and the chest X-ray classfication web app.

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data will be kept [here](https://github.com/pavan-kalyan/chest-xray-classification-azure/tree/master/data/raw) within this repo. It can be downloaded by running src/get_and_extract.py 
or by running notebook/get_and_extract.ipynb
3. For data processing and preparation, run notebook/1_data_prep.ipynb or src/data_prep.ipynb. This will process the dataset and store the train/validate/test split in data/processed/.
4. (a) For training the model on Azure's compute target. Make a Machine learning workspace on Azure and download the config.json file to notshared/ also make a storage account on Azure.
Then run the 2_pytorch train notebook, this will setup the experiment and automatically procure a compute target and train the model there and store the output on the Azure workspace.
4. (b) For training the model on your local system or a VM (e.g [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-in/services/virtual-machines/data-science-virtual-machines/)), simply run the following
    ```
    python src/pytorch_train.py --data-dir="data/processed/images" 
    ```
5. For Deploying the model on Azure, run the notebook 3_pytorch_run.py. This will register the model,create the image and deploy on Azure Container Instances. The link to the webapp will be available on the Azure portal for the deployment.

## Disclaimer
This x-ray image pathology classification system is neither intended for real clinical diagnosis nor to be used as a substitute for a medical professional. The performance of this model for clinical use has not been established.


## Acknowledgements
With deep gratitude to researchers and developers at PyTorch, NIH, Stanford, Project Jupyter and [AzureChestXray repo](https://github.com/Azure/AzureChestXRay). With special thanks to my mentors at Microsoft for their guidance.
