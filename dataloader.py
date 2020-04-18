import wfdb
import os
import pandas as pd
import torch
import numpy as np
from scipy import signal


def get_midata():
    
    dbpath = "C:/CodeBank/Other/MI/database/ptb/"

    db = pd.DataFrame()

    with open(dbpath+"CONTROLS", 'r') as f:
        controls = f.readlines()
        
        controls = [subj.rstrip('\n') for subj in controls]


    patients = os.listdir(dbpath)

    for patient in patients:
        try:
            files = os.listdir(dbpath+patient)
            for _file in files:
                if '.dat' in _file:
                    waveform_path = patient+"/"+_file
                    waveform_path = waveform_path[:-4]
                    if waveform_path not in controls:
                        signals = wfdb.rdsamp(dbpath + waveform_path)
                        comments = signals[1]['comments']
                        fs = signals[1]['fs']
                        signals = signals[0]
                        l1 = signals[:,0]
                        l2 = signals[:,1]
                        # v1 = signals[:,6]
                        # v6 = signals[:,-1]

                        label = pull_label(comments)

                        # datapoint = (label, l2, v1, v6)
                        datapoint = (label, l1, l2)
                        data = augment(datapoint, fs)
                        db = db.append(data, ignore_index=True)

        except NotADirectoryError:
            print(patient, "not a dir")

    db.to_pickle("./MIwaveforms.csv")

def get_bihdata():

    dbpath = "C:/CodeBank/Other/MI/database/mitbih/"

    db = pd.DataFrame()

    patients = os.listdir(dbpath)

    for patient in patients:
        try:
            if '.dat' in patient:
                waveform_path = patient[:-4]
                signals = wfdb.rdsamp(dbpath + waveform_path)
                comments = signals[1]['comments']
                fs = signals[1]['fs']
                signals = signals[0]
                l1 = signals[:,0]
                l2 = signals[:,1]

                label = pull_label(comments)

                datapoint = (label, l1, l2)
                data = augment(datapoint, fs)
                db = db.append(data, ignore_index=True)

        except NotADirectoryError:
            print(patient, "not a dir")

    db.to_pickle("./Regwaveforms.csv")

def pull_label(comments):

    for comment in comments:
        comment = comment.lower()
        if 'myocardial infarction' in comment or 'mi' in comment:
            return 1
        else:
            continue
    return 0

def augment(datapoint, fs):

    crop = 4 * fs

    label = datapoint[0]

    outputs = pd.DataFrame()

    thresh1 = datapoint[1].max()*0.07
    thresh2 = datapoint[2].max()*0.07

    for i in range(len(datapoint[1])//crop):

        li = datapoint[1][i*crop:(i+1)*crop]
        lii = datapoint[2][i*crop:(i+1)*crop]

        if (abs(li).max() < thresh1 or abs(lii).max() < thresh2):
            break

        new_datapoint = [{'label':label, 'LI': signal.resample(li, 512), 'LII':signal.resample(lii, 512)}]

        df = pd.DataFrame(new_datapoint)
        outputs = outputs.append(df, ignore_index=True)

    return outputs

def import_data(dbpath):
    df = pd.read_pickle(dbpath)
    return df

def prepare_data():
    
    np.random.seed(7)

    db1 = pd.read_pickle("./database/MIwaveforms.csv")
    db2 = pd.read_pickle("./database/Regwaveforms.csv")

    db2 = db2.sample(len(db1))

    db1 = db1.append(db2, ignore_index=True)

    db1 = db1.sample(frac=1).reset_index(drop=True)

    labels = db1.iloc[:,0]

    l1 = db1.iloc[:,1].apply(pd.Series)

    l2 = db1.iloc[:,2].apply(pd.Series)

    labels.to_pickle('./Labels.csv')
    l1.to_pickle('./L1.csv')
    l2.to_pickle('./L2.csv')


def split_data():

    labels = pd.read_pickle('./Labels.csv')
    l1 = pd.read_pickle('./L1.csv')
    l2 = pd.read_pickle('./L2.csv')

    labels = labels.values
    l1 = l1.values
    l2 = l2.values

    data = []

    for i in range(len(l1)):
        l1_i = torch.tensor(l1[i])
        l2_i = torch.tensor(l2[i])
        data.append(torch.stack((l1_i,l2_i)).transpose(1,0))

    split = len(labels)//10

    # data is already randomized

    training = ( data[:split*8], labels[:split*8])

    validation = (data[split*8:split*9], labels[split*8:split*9])

    test = (data[split*9:], labels[split*9:])
    
    return {'Training': training, 'Validation': validation, 'Test': test}

