from typing import Type
import numpy as np 
import pandas as pd   
import pickle

from rdkit import Chem 
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np
from tqdm import tqdm

def model_predict(feat,Env = 'SIF'):
    SIF_FEATURE_LIST = ['MinAbsEStateIndex','qed','MinPartialCharge','Chi1v','PEOE_VSA8','SMR_VSA10','SMR_VSA4','SMR_VSA6','SlogP_VSA3','EState_VSA10','EState_VSA2','EState_VSA6','EState_VSA8','EState_VSA9','VSA_EState1','VSA_EState4','VSA_EState8']
    SGF_FEATURE_LIST = ['ExactMolWt','NumHAcceptors','NumHDonors','MolLogP','TPSA','NumRotatableBonds']

    if Env == 'SIF':
        print('Prediting SIF Stability...')
        feat = feat[SIF_FEATURE_LIST]
        df_pred=feat.assign(Env='Intestinal')
        model = pickle.load(open('model/SIF_model', 'rb'))
    elif Env == 'SGF':
        feat = feat[SGF_FEATURE_LIST]
        print('Prediting SGF Stability...')
        df_pred=feat.assign(Env='Gastric')
        model = pickle.load(open('model/SGF_model', 'rb'))
    else:
        raise KeyError('Wrong Env Set, should be either SIF or SGF')
    

    pred_Env = model['GI_encoder'].transform(np.array(df_pred['Env']).reshape(-1, 1))
    pred_features=feat

    #PCA
    pred_features = model['feature_scaler'].transform(np.array(pred_features))
    pred_Features = np.concatenate([pred_Env,pred_features],axis=1)

    y_pred = model['clf'].predict(pred_Features)
    y_pred = model['Label_encoder'].inverse_transform(y_pred.reshape(-1, 1))
    return (y_pred)

def pep_feat(SMILES_list_PATH):
    pep_db = pd.read_csv(SMILES_list_PATH)
    des_list = [x[0] for x in Descriptors._descList]
    feat=np.zeros([len(pep_db['SMILES']),len(des_list)])
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)

    for i in tqdm(range(len(pep_db))):
        mol = Chem.MolFromSmiles(pep_db['SMILES'][i])
        feat[i] = calculator.CalcDescriptors(mol)

    return pd.DataFrame(feat,columns=des_list)

def save_results(SMILES_list_PATH,SIF_Stability,SGF_Stability):
    pep_db = pd.read_csv(SMILES_list_PATH)
    pep_db=pep_db.assign(Stability_in_SIF=SIF_Stability)
    pep_db=pep_db.assign(Stability_in_SGF=SGF_Stability)
    pep_db.to_csv(SMILES_list_PATH,index=False)
    print('Predicted SIF/SGF stability saved to the original file: ',SMILES_list_PATH)
    return pep_db