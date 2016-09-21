#!/usr/bin/python2.7

import re
import os
import shutil
import numpy as np
import pandas as pd
import scipy
import datetime
from scipy.io import savemat, loadmat
from functools import partial
from multiprocessing import Pool 
from sklearn.preprocessing import normalize
from scipy.signal import correlate, resample, coherence
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
#import xgboost as xgb
from sklearn.metrics import make_scorer
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, freqz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, f1_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

class XGBoostClassifier():
    def __init__(self, num_boost_round=100, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self             

def getFiles(directory, extension='.mat'):
    # Sort all the files of the directory
    filenames = sorted(os.listdir(directory))

    # Obtain all the files of the directory
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]

    # Return all the files of the directory
    return files_with_extension

def createSubmission(prediction, MatTestingFiles):
    matNumber = 0
    
    # Make submission file
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('File,Class\n')

    # Obtain header
    for MatFile in MatTestingFiles:
        strToWrite = str(os.path.basename(MatFile)) + ',' + str(prediction[matNumber]) + '\n'
        f.write(strToWrite)
        matNumber += 1       
    f.close()

def computeFinalScore(dataLen, MatTestingFiles):
    matPrediction = []
    
    # Read data from HDFStore file 
    X_train = pd.read_hdf('trainingDataT1.h5', 'data', stop = dataLen*500)
    y_train = pd.read_hdf('trainingDataT1.h5', 'y', stop = dataLen*500)

    print 'X and y train read'
    
    X_train = X_train.as_matrix()
    y_train = np.array(y_train).ravel()
    y_train = map(int, y_train)

    print 'training files processed'
                
    X_test = pd.read_hdf('testingDataT1.h5', 'data')
    X_test = X_test.as_matrix()
    print X_test[0:18000].shape

    print 'testing files processed'

    print 'Data loaded'
    '''
    # Create GBT algorithm with xgboost library
    clf = XGBoostClassifier(
        objective = 'binary:logistic',
        booster = 'gbtree',
        eval_metric = 'auc',
        tree_method = 'exact',
        num_class = 2,
        silent = 1,
        seed = 42,
        )
    
    parameters = {
        'eta': [0.015, 0.025],#[0.01, 0.015, 0.025, 0.05, 0.1],
        'gamma': [0.1, 0.5],#[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'max_depth': [5, 9],#[3, 5, 7, 9, 12, 15, 17, 25],
        'min_child_weight': [3, 7],#[1, 3, 5, 7],
        'subsample': [0.7, 0.9],#[0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9],#[0.6, 0.7, 0.8, 0.9, 1.0],
        'lambda': [0.1, 1.0],#[0.05, 0.1, 1.0],
        'alpha': [0.1, 1.0],#[0, 0.1, 0.5, 1.0],
    }
    '''
    clf = RandomForestClassifier(random_state = 42, n_jobs=8)
    '''
    parameters = {
        'max_depth': [2, 3, 4],
        'loss': ['deviance', 'exponential'],
    }
    
    eval_size = 0.10
    kf = StratifiedKFold(y_train, round(1. / eval_size), shuffle=True, random_state=42)

    scoring_fnc = make_scorer(f1_score)

    clf = GridSearchCV(clf, parameters, scoring_fnc, cv=kf, n_jobs=-1)
    '''
    clf.fit(X_train, y_train)
    '''
    clf = clf.best_estimator_

    print 'best param for max_depth : ', clf.get_params()['max_depth']
    print 'best param for loss : ', clf.get_params()['loss']
    '''        
    if hasattr(clf, "predict_proba"):
        prediction = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prediction = clf.decision_function(X_test)
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

    scale = 0
    for matFile in range(len(MatTestingFiles)):
        matPrediction.append(np.mean(prediction[scale:scale+dataLen]))
        if matPrediction[matFile] > 0.5:
            matPrediction[matFile] = np.amax(prediction[scale:scale+dataLen])
        elif matPrediction[matFile] < 0.5:
            matPrediction[matFile] = np.amin(prediction[scale:scale+dataLen]) 
        scale += dataLen

    print 'Creating submission file'

    createSubmission(matPrediction, MatTestingFiles)       	   


def computeTestScore():           
    # Read data from HDFStore file
    X = pd.read_hdf('trainingDataT3.h5', 'data')
    y = pd.read_hdf('trainingDataT3.h5', 'y')

    print 'X and y read'
    
    X = X.as_matrix()
    y = np.array(y).ravel()
    y = map(int, y)

    print 'X and y processed'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = RandomForestClassifier(random_state = 42, n_jobs=8)
    clf.fit(X_train, y_train) 

    prob_pos_clf = clf.predict_proba(X_test)[:, 1]

    '''
    # Model with isotonic calibration
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

    # Gaussian Naive-Bayes with sigmoid calibration
    clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
    clf_sigmoid.fit(X_train, y_train)
    prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

    print("Brier scores: (the smaller the better)")

    clf_score = brier_score_loss(y_test, prob_pos_clf, sample_test)
    print("No calibration: %1.3f" % clf_score)

    clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sample_test)
    print("With isotonic calibration: %1.3f" % clf_isotonic_score)

    clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sample_test)
    print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

    print("AUC scores:") 
    '''
    clf_auc_score = roc_auc_score(y_test, prob_pos_clf)
    print("No calibration: %1.3f" % clf_auc_score)
    '''
    clf_isotonic_auc_score = roc_auc_score(y_test, prob_pos_isotonic)
    print("With isotonic calibration: %1.3f" % clf_isotonic_auc_score)

    clf_sigmoid_auc_score = roc_auc_score(y_test, prob_pos_sigmoid)
    print("With sigmoid calibration: %1.3f" % clf_sigmoid_auc_score)
    '''
def processData(MatFiles, dataType):
    fileNumber = 0
    dataLenght = 0
    init = 0

    if dataType == 'training':  
        hf = pd.HDFStore('trainingDataT3test.h5', mode='w')
    else:
        hf = pd.HDFStore('testingDataT3test.h5', mode='w')
    
    # Obtain data
    for MatFile in MatFiles:
        data = None
        y = None 
        try:
            d = loadmat(MatFile)
        except ValueError as ex:
            print(u'Error loading MAT file {}: {}'.format(os.path.basename (MatFile), str(ex)))
            continue

        # Extract each data from the file
        datastruct = d['dataStruct']
        channelIndices = datastruct['channelIndices'][0][0][0]
        iEEGsamplingRate = datastruct['iEEGsamplingRate'][0][0][0][0]
        nSamplesSegment = datastruct['nSamplesSegment'][0][0][0][0]
        ieegData = datastruct['data'][0][0].transpose()

        segmentEnd = 30

        # Extract labels
        if dataType == 'training': 
            # extract the class of the file
            header = re.split(r"[_.]", os.path.basename (MatFile))
            category = header[2]
            subY = np.repeat(category, nSamplesSegment/(iEEGsamplingRate/segmentEnd), axis=0)
            subY = pd.DataFrame({'y': subY})

        # Frequency vector
        f = iEEGsamplingRate*np.linspace(0,nSamplesSegment/(iEEGsamplingRate/segmentEnd),nSamplesSegment/(iEEGsamplingRate/segmentEnd))/nSamplesSegment                 

        nElectrodes = len(channelIndices)
        
        for el in range(nElectrodes):

            Y = np.fft.fft(ieegData[el])

            filtered = []
            b= []               # store filter coefficient
            cutoff = [0.001, 0.5, 3.0, 8.0, 12.0, 30.0]

            for band in xrange(0, len(cutoff)-1):
                wl = 2*cutoff[band]/iEEGsamplingRate*np.pi
                wh = 2*cutoff[band+1]/iEEGsamplingRate*np.pi
                M = 512      # Set number of weights as 128
                bn = np.zeros(M)

                for i in xrange(0,M):     # Generate bandpass weighting function
                    n = i - M/2       # Make symmetrical
                    if n == 0:
                        bn[i] = wh/np.pi - wl/np.pi;
                    else:
                        bn[i] = (np.sin(wh*n))/(np.pi*n) - (np.sin(wl*n))/(np.pi*n)   # Filter impulse response
                
                
                bn = bn*np.kaiser(M,5.2)  # apply Kaiser window, alpha= 5.2
                b.append(bn)

                [w,h]=freqz(bn,1)
                filtered.append(np.convolve(bn, ieegData[el])) # filter the signal by convolving the signal with filter coefficients
            '''
            plt.subplot(16, 1, el+1)
            plt.plot(ieegData[el])
            for i in xrange(0, len(filtered)):
                y_p = filtered[i]
                plt.plot(y_p[ M/2:nSamplesSegment+M/2])
            plt.axis('tight')
            plt.xlabel('Time (seconds)')
            '''
            #ax = plt.subplot(nElectrodes, 1, el+1)
            #plt.plot(f,2*np.abs(Y[0:nSamplesSegment/(iEEGsamplingRate/segmentEnd)]))
            for i in xrange(0, len(filtered)):
                Y = filtered[i]
                Y = np.fft.fft(Y[M/2:nSamplesSegment+M/2])
                subData = abs(Y[0:nSamplesSegment/(iEEGsamplingRate/segmentEnd)])
                subFrame = pd.DataFrame({'%d_%d' % (el, i): subData})
                if data is None:
                    data = subFrame
                else:
                    data = data.join(subFrame)
                    
                #plt.plot(f,subData)
                #plt.axis('tight')
                #plt.axis([0, 30, 0, 300000])
                #ax.set_autoscale_on(False)
        #print data.shape     
        #plt.legend(['infra band, 0.0-0.5 Hz', 'delta band, 0.5-3 Hz','theta band, 3-8 Hz','alpha band, 8-12 Hz','beta band, 12-30 Hz'])
            
        #plt.show()         

        # Write in HDFStore file the processed data
        if init == 0:
            hf.put('data', data, format='table')
            dataLenght = data.shape[0]
            print data.shape
        else:
            hf.append('data', data, format='table')
        
        if dataType == 'training':
            if init == 0:
                hf.put('y', subY, format='table')
            else:
                hf.append('y', subY, format='table')
        
        fileNumber += 1
        init = 1
        
        if fileNumber % 10 == 0.0:
            print '%d files done' % fileNumber
            
    hf.close()
    
    if dataType == 'training':  
        print 'Training done'
    else:
        print 'Testing done' 

    return fileNumber, dataLenght


if __name__=="__main__":
    #MatTrainingFiles= getFiles("train_2")
    MatTestingFiles= getFiles("test_1")
    #NumberTrainingFiles, dataTrainingLenght = processData(MatTrainingFiles, 'training')
    #NumberTestingFiles, dataTestingLenght = processData(MatTestingFiles, 'testing')

    computeFinalScore(18000, MatTestingFiles)
    #computeTestScore()

