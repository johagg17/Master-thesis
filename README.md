# Modelling patient trajectories
This is a Master's thesis for modelling patient trajectories using [Transformers](https://arxiv.org/pdf/1706.03762.pdf) at Halmstad University togehter with my collegue and friend [Linus Lerjebo](https://github.com/Lerjebo), and our supervisiors.\
There exists some novel methods which have shown on new state-of-art results for modelling patient trajectories, some of them are highlighted in the table below:

| Method  | Description | Paper |
| ------------- | ------------- | ------------- |
| BEHRT | Training the BERT model on EHR data. Given a patient's medical history, predict future diseases. As input, BEHRT add an additional feature apart from orignial BERT, age embedding. | [Link](https://www.nature.com/articles/s41598-020-62922-y)|
| ClinicalBERT  | ClinicalBERT trains BERT on clinical notes in order to predict if a patient should be readmitted or not.   | [Link](https://github.com/kexinhuang12345/clinicalBERT) |
| G-BERT  | Combines graph neural networks with BERT for medication recommendation. The medical onotology between medical codes is extracted from their GNN, and then fed into the BERT model.   | [Link](https://arxiv.org/abs/1906.00346) |
| Hi-BEHRT| Use BEHRT model to implement an hierarchy using a sliding window, this for capturing long-term dependency. The original BEHRT can only capture sentences of token length 512, while Hi-BEHRT can train on token length 1220. | [Link](https://arxiv.org/abs/2106.11360)|

## Project description
In this project, we are combining the Hi-BEHRT and G-BERT, for both modelling for multimodal transformers (text and numerical inputs), capture long-term dependencies, and learn the hierarchical medical ontology of medical codes. As ouput of our model, sequence classification. Given a patient's medical history, can we predict future heart failure patients, and different patient attributes (medication recommendation, readmission, etc...).  
## Dataset
For this project we use [MIMIC-IV](https://physionet.org/content/mimiciv/0.4/) dataset, which contains data for over 60,000 patients. \
The dataset is accessible by taking a course at [CITI](https://about.citiprogram.org/courses/?reset=true), and here is a more in-depth guide for accessing [MIMIC dataset](https://towardsdatascience.com/getting-access-to-mimic-iii-hospital-database-for-data-science-projects-791813feb735), though this guide covers MIMIC-III, but the steps for accessing MIMIC-IV should be the same as accessing MIMIC-III. 

## Requirements


## How to
