# Using Pretrained Machine Learning Models to Predict Peptide Stability Profile in Simulated Gastric/Intestinal Fluids

Publication: *DOI:*

Environment: Python 3.7.7

Dependancies:
- scikit-learn: 0.24.2
- py-xgboost: 1.3.3
- rdkit: 2020.03.3.0
- pandas: 1.3.0
- numpy: 1.20.3

# To Predict

In order to predict peptide stability, the structure of the peptide, represented in *isomeric SMILES* notation, should be prepared first.

1. Edit the .csv file in the folder to fill in peptide information (Multiple prediction are supported by adding extra rows). The last two columns 'Stability_in_SIF' and 'Stability_in_SGF' can be left empty and will be filled automatically.

2. Run the code in the jupyter notebook.

3. The result will be displayed on the notebook and also saved into the .csv file.


