import wfdb
import os
import pandas as pd
import torch
import numpy as np
from scipy import signal
import scipy.io
import preprocess


def get_midata():
    
    dbpath = "C:/CodeBank/Other/MI/midetect/database/ptb/"

    db = pd.DataFrame()

    # ignore controls as they are confusing to the classifier
    # not necessarily T or F

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
                        l2 = signals[:,1]
                        v1 = signals[:,6]
                        v6 = signals[:,-1]

                        label = pull_label(comments)

                        datapoint = (label, l2, v1, v6)
                        data = augment(datapoint, fs)
                        db = db.append(data, ignore_index=True)

        except NotADirectoryError:
            print(patient, "not a dir")

    db.to_pickle("./database/MIwaveforms.csv")

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

def get_cpscdata():

    dbpath = "C:/CodeBank/Other/MI/midetect/database/cpsc/"

    db = pd.DataFrame()

    reference = pd.read_csv(dbpath+'REFERENCE.csv')

    for k in range(len(reference)):

        patient = reference.iloc[k,0]

        labels = reference.iloc[k,1:]

        # 7 = ST depression
        # 8 = ST elevation
        if 7 in labels.values: #or 8 in labels.values:
            label = 1
        else:
            label = 0

        mat = scipy.io.loadmat(dbpath+patient+'.mat')

        signals = mat['ECG'][0][0][2]

        l2 = signals[2,:]
        v1 = signals[6,:]
        v6 = signals[-1,:]

        datapoint = (label, l2, v1, v6)
        data = augment(datapoint, 500)
        db = db.append(data, ignore_index=True)

    db.to_pickle("./database/CPSC.csv")

def pull_label(comments):

    # Label a patient's signals based on the doctor's diagnosis

    for comment in comments:
        comment = comment.lower()
        if 'myocardial infarction' in comment or 'mi' in comment:
            return 1
        else:
            continue
    return 0

def augment(datapoint, fs):

    # Turn a variable length ECG into 4 second pieces 
    # Resample 4 second ECG to 500Hz

    crop = 4 * fs

    label = datapoint[0]

    outputs = pd.DataFrame()

    # Sometimes ECG has dead signal at the edges
    # Toss broken ECG pieces based on thresholding

    thresh1 = datapoint[1].max()*0.07
    thresh2 = datapoint[2].max()*0.07
    thresh3 = datapoint[3].max()*0.07

    for i in range(len(datapoint[1])//crop):

        lii = datapoint[1][i*crop:(i+1)*crop]
        v1 = datapoint[2][i*crop:(i+1)*crop]
        v6 = datapoint[3][i*crop:(i+1)*crop]

        if (abs(lii).max() < thresh1 or abs(v1).max() < thresh2 or abs(v6).max() < thresh3):
            break
        
        # Get the morlet wavelet coefficients for 0.8-1 Hz
        wvlt = preprocess.wavelet(lii)

        new_datapoint = [{'label':label, 'LII': signal.resample(lii, 2000), 'V1':signal.resample(v1, 2000), 'V6':signal.resample(v6, 2000), 'Wavelets':wvlt}]

        df = pd.DataFrame(new_datapoint)
        outputs = outputs.append(df, ignore_index=True)

    return outputs

def import_data(dbpath):
    df = pd.read_pickle(dbpath)
    return df

def mix_dbs():

    # Use this function for integration of a second database into set

    db1 = pd.read_pickle("./database/MIwaveforms.csv")
    db2 = pd.read_pickle("./database/CPSC.csv")

    # db1 all True
    # db2 mixed
    # create 50-50 split

    db1 = db1.append(db2[db2['label']==1])

    all_false = db2[db2['label']==0]

    all_false = all_false.sample(len(db1))

    db1 = db1.append(all_false, ignore_index=True)

    # Shuffle

    db1 = db1.sample(frac=1).reset_index(drop=True)

    # Split to smaller files

    labels = db1.iloc[:,0]

    l2 = db1.iloc[:,1].apply(pd.Series)

    v1 = db1.iloc[:,2].apply(pd.Series)

    v6 = db1.iloc[:,3].apply(pd.Series)

    labels.to_pickle('./database/Labels.csv')
    l2.to_pickle('./database/L2.csv')
    v1.to_pickle('./database/V1.csv')
    v6.to_pickle('./database/V6.csv')

def prepare_data():

    db = pd.read_pickle("./database/CPSC.csv")

    trues = db[db['label']==1]

    falses = db[db['label']==0]

    db1 = trues.append(falses.sample(len(trues)), ignore_index=True)

    # Shuffle

    db1 = db1.sample(frac=1).reset_index(drop=True)

    # Split to smaller files

    labels = db1.iloc[:,0]

    l2 = db1.iloc[:,1].apply(pd.Series)

    v1 = db1.iloc[:,2].apply(pd.Series)

    v6 = db1.iloc[:,3].apply(pd.Series)

    wvlt = db1.iloc[:,4].apply(pd.Series)

    labels.to_pickle('./database/Labels.csv')
    l2.to_pickle('./database/L2.csv')
    v1.to_pickle('./database/V1.csv')
    v6.to_pickle('./database/V6.csv')
    wvlt.to_pickle('./database/wvlt.csv')

def split_data():

    labels = pd.read_pickle('./database/Labels.csv')
    l2 = pd.read_pickle('./database/L2.csv')
    v1 = pd.read_pickle('./database/V1.csv')
    v6 = pd.read_pickle('./database/V6.csv')
    wvlt = pd.read_pickle('./database/wvlt.csv')

    labels = labels.values
    l2 = l2.values
    v1 = v1.values
    v6 = v6.values
    wvlt = wvlt.values

    data = []

    # data format:
    # list(tensor([Seq Len, Channels]))

    for i in range(len(l2)):
        l2_i = torch.tensor(l2[i])
        v1_i = torch.tensor(v1[i])
        v6_i = torch.tensor(v6[i])
        wvlt_i = torch.tensor(wvlt[i])
        data.append(torch.stack((l2_i, v1_i, v6_i, wvlt_i)).transpose(1,0))

    split = len(labels)//10

    # data is already randomized

    training = ( data[:split*6], labels[:split*6])

    validation = (data[split*6:split*8], labels[split*6:split*8])

    test = (data[split*8:], labels[split*8:])
    
    return {'Training': training, 'Validation': validation, 'Test': test}

