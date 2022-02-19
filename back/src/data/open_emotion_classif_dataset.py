import pandas as pd
import torch
import os

    
def open_emotion_classif_dataset(dataset_type, tsv_filename):
    """
    returns a pytorch dataset from a tsv file located in ../../res/data/processed/emotion_classif/ of a different type ('train', 'dev' or 'test')
    Warning: only use *_french2 tsv files
    """
    valid = {'train','dev','test'}
    if dataset_type not in valid:
        raise ValueError("Value of dataset_type parameter must be either 'train', 'dev' or 'test'.")
    data_dirname = os.path.dirname(__file__)
    data_path = os.path.join(
        data_dirname, 
        '../../res/data/processed/emotion_classif/'
        + dataset_type 
        + '/'
        + tsv_filename
    )
    df = pd.read_csv(data_path, sep="\t", lineterminator="\n", header=None)
    df.columns = ['text', 'label', 'other']
    labels_df = df.label.str.get_dummies(sep=",")
    labels_tensor = torch.tensor(labels_df.values).type(torch.FloatTensor)
    sentences_list = list(df.text)
    return sentences_list, labels_tensor



