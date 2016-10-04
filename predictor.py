#!/usr/bin/python2.7

import re
import os
import shutil
import numpy as np
import pandas as pd
import scipy
import datetime
import time
from scipy.io import savemat, loadmat
from functools import partial
from multiprocessing import Pool 
from sklearn.preprocessing import normalize
from scipy.signal import correlate, resample, coherence
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from sklearn.metrics import make_scorer
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, freqz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from collections import deque
import pywt

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
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '_test300' + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('File,Class\n')

    # Obtain header
    for MatFile in MatTestingFiles:
        strToWrite = str(os.path.basename(MatFile)) + ',' + str(prediction[matNumber]) + '\n'
        f.write(strToWrite)
        matNumber += 1       
    f.close()

def computeFinalScore(MatTestingFiles):
    matPrediction = []
    
    # Read data from HDFStore file 
    X_train = pd.read_hdf('trainingDataT1.h5', 'data')
    y_train = pd.read_hdf('trainingDataT1.h5', 'y')

    print 'X and y train read'
    X_train.drop(X_train.columns[[1,2,3]], axis=1, inplace=True)
    X_train = X_train.as_matrix()
    y_train = np.array(y_train).ravel()
    y_train = map(int, y_train)

    for i in range(20):
        print X_train[i]
        print y_train[i]

    print 'training files processed'
                
    X_test = pd.read_hdf('testingDataT1300Test.h5', 'data')
    X_test.drop(X_test.columns[[1]], axis=1, inplace=True)
    X_test = X_test.as_matrix()   

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
    clf = GradientBoostingClassifier(random_state = 42)
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
    
    # Model with sigmoid calibration
    clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_sigmoid.fit(X_train, y_train)
    '''
    clf = clf.best_estimator_
    
    print 'best param for max_depth : ', clf.get_params()['max_depth']
    print 'best param for loss : ', clf.get_params()['loss']
    '''        
    if hasattr(clf, "predict_proba"):
        prediction = clf_sigmoid.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prediction = clf.decision_function(X_test)
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    '''
    scale = 0
    for matFile in range(len(MatTestingFiles)):
        matPrediction.append(np.mean(prediction[scale:scale+dataLen]))
    
        if matPrediction[matFile] > 0.5:
            matPrediction[matFile] = np.amax(prediction[scale:scale+dataLen])
        elif matPrediction[matFile] < 0.5:
            matPrediction[matFile] = np.amin(prediction[scale:scale+dataLen])
           
        scale += dataLen
    '''
    print 'Creating submission file'

    createSubmission(prediction, MatTestingFiles)       	   


def computeTestScore():           
    # Read data from HDFStore file
    X = pd.read_hdf('trainingDataT1.h5', 'data')
    y = pd.read_hdf('trainingDataT1.h5', 'y')

    print 'X and y read'

    X.drop(X.columns[[20,21,22]], axis=1, inplace=True)
    print X.shape
    X = X.as_matrix()
    y = np.array(y).ravel()
    y = map(int, y)

    listX0 = []
    listX1 = []
    '''
    for i in range(X.shape[0]):
        if y[i] == 1.0:
            listX1.append(X[i][0])
        elif y[i] == 0.0:
            listX0.append(X[i][0])
    
    print 'min = %d, max=%d' % (np.amin(listX0), np.amax(listX0))
    print np.median(listX0)
    print 'min = %d, max=%d' % (np.amin(listX1), np.amax(listX1))
    print np.median(listX1)
    print 'ratio : ', np.median(listX1)/np.median(listX0)
    '''
    '''
    print 'min = %d, max=%d' % (np.amin(X), np.amax(X))
    print np.median(X)
    print 'X and y processed'
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #clf = GradientBoostingClassifier(random_state = 42)
    
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
        'eta': [0.01],#[0.01, 0.015, 0.025, 0.05, 0.1],
        'gamma': [0.1],#[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'max_depth': [3],#[3, 5, 7, 9, 12, 15, 17, 25],
        'min_child_weight': [3],#[1, 3, 5, 7],
        'subsample': [0.7],#[0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [1.0],#[0.6, 0.7, 0.8, 0.9, 1.0],
        'lambda': [0.1],#[0.05, 0.1, 1.0],
        'alpha': [0.01],#[0, 0.1, 0.5, 1.0],
    }

    eval_size = 0.10
    kf = StratifiedKFold(y_train, round(1. / eval_size), shuffle=True, random_state=42)

    scoring_fnc = make_scorer(roc_auc_score)

    clf = GridSearchCV(clf, parameters, scoring_fnc, cv=kf, n_jobs=-1)
    
    clf.fit(X_train, y_train)
    
    clf = clf.best_estimator_
    
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]
   
    # Model with isotonic calibration
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

    # Model with sigmoid calibration
    clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
    clf_sigmoid.fit(X_train, y_train)
    prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

    print("Brier scores: (the smaller the better)")

    clf_score = brier_score_loss(y_test, prob_pos_clf)
    print("No calibration: %1.3f" % clf_score)

    clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic)
    print("With isotonic calibration: %1.3f" % clf_isotonic_score)

    clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid)
    print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

    print("AUC scores:") 
    
    clf_auc_score = roc_auc_score(y_test, prob_pos_clf)
    print("No calibration: %1.3f" % clf_auc_score)
    
    clf_isotonic_auc_score = roc_auc_score(y_test, prob_pos_isotonic)
    print("With isotonic calibration: %1.3f" % clf_isotonic_auc_score)

    clf_sigmoid_auc_score = roc_auc_score(y_test, prob_pos_sigmoid)
    print("With sigmoid calibration: %1.3f" % clf_sigmoid_auc_score)

def squareList(list):
    return map(lambda x: x ** 2, list)

def processData(MatFiles, dataType):
    fileNumber = 0
    dataLenght = 0
    init = 0

    if dataType == 'training':  
        hf = pd.HDFStore('trainingDataT1.h5', mode='w')
    else:
        hf = pd.HDFStore('testingDataT1.h5', mode='w')

    # Obtain data
    for MatFile in MatFiles:
        y = []
        cleanedList = []
        frequencyList = []
        dataList = []
        nSamplesSegmentList = []
        totalDropOut = 0

        energyListGlob = []
        statisticGlobList = []
        
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

        
        cutoff = [0.001, 0.5, 3.0, 7.0, 12.0, 30.0]
        segmentEnd = cutoff[-1]
        
        # Extract labels
        if dataType == 'training': 
            # extract the class of the file
            header = re.split(r"[_.]", os.path.basename (MatFile))
            category = header[2]
            y.append(category)
            y = pd.DataFrame({'y': y})

        # Frequency vector
        #f = iEEGsamplingRate*np.linspace(0,nSamplesSegment/(iEEGsamplingRate/segmentEnd),nSamplesSegment/(iEEGsamplingRate/segmentEnd))/nSamplesSegment                 
        
        nElectrodes = len(channelIndices)
        
        for el in range(nElectrodes):      
            dequeList = deque(ieegData[el])
            for i in range(nSamplesSegment):
                if dequeList[0] == 0 and dequeList[0] < 300:
                    dequeList.popleft()
                else:
                    dequeList.rotate(-1)

            cleanedList.append(dequeList)        
            nSamplesSegmentList.append(len(dequeList))

        nSamplesSegment = int(np.amin(nSamplesSegmentList)/100)*100
        
        energyList = []
        statisticList = []
        
        for el in range(nElectrodes):
            maxList = []
            statListEl = []

            minValue = np.amin(ieegData[el])
            statListEl.append(minValue)
            maxValue = np.amax(ieegData[el])
            statListEl.append(maxValue)
            meanValue = np.mean(ieegData[el])
            statListEl.append(meanValue)
            medianValue = np.median(ieegData[el])
            statListEl.append(medianValue)
            stdValue = np.std(ieegData[el])
            statListEl.append(stdValue)
            varValue = np.var(ieegData[el])
            statListEl.append(varValue)

            statisticList.append(statListEl)

            coeffs = pywt.wavedec(ieegData[el], 'db1', level=17)

            #f = iEEGsamplingRate*np.linspace(0,nSamplesSegment/(iEEGsamplingRate/segmentEnd),nSamplesSegment/(iEEGsamplingRate/segmentEnd))/nSamplesSegment

            for i in range(len(coeffs)):
                energy = np.sum(squareList(coeffs[i]))
                maxList.append(energy)
                
            energyList.append(maxList)

            ieegDataCleaned = np.array(cleanedList[el])
            
            if len(ieegDataCleaned) < 100 and dataType == 'training':
                totalDropOut = 1
                print 'too much data drop outs : ', os.path.basename(MatFile)
                break

            if len(ieegDataCleaned) < 100:
                for el in range(nElectrodes):
                    maxList = []
                    for i in range(len(cutoff)-1):
                        maxList.append(0.0)
                    frequencyList.append(maxList)
                break               

            filtered = []
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
                filtered.append(np.convolve(bn, ieegDataCleaned[0:nSamplesSegment])) # filter the signal by convolving the signal with filter coefficients
            '''
            plt.subplot(16, 1, el+1)
            plt.plot(ieegData[el])
            #for i in xrange(0, len(filtered)):
                #y_p = filtered[i]
                #plt.plot(y_p[ M/2:nSamplesSegment+M/2])
            plt.axis('tight')
            plt.xlabel('Time (seconds)')
            plt.show()
            '''
            
            #ax = plt.subplot(nElectrodes, 1, el+1)
            #plt.plot(f,2*np.abs(Y[0:nSamplesSegment/(iEEGsamplingRate/segmentEnd)]))
            for i in xrange(0, len(filtered)):
                Y = filtered[i]
                Y = np.fft.fft(Y[int(M/2):int(nSamplesSegment)+int(M/2)])
                subData = abs(Y[0:int(nSamplesSegment/(iEEGsamplingRate/segmentEnd))])
                maxList.append(np.median(subData))         
                #plt.plot(f,subData)
                #plt.axis('tight')
                #plt.axis([0, 30, 0, 300000])
                #ax.set_autoscale_on(False)
                
            frequencyList.append(maxList)

        #plt.legend(['delta band, 0.5-3 Hz','theta band, 3-8 Hz','alpha band, 8-12 Hz','beta band, 12-30 Hz'])

        #print category    
        #plt.show()
        
        if totalDropOut != 1:
            # Gather information on statistics
            dataStatistics = pd.DataFrame(statisticList)

            for stat in range(len(statListEl)):
                statisticGlobList.append(np.median(dataStatistics[stat]))

            dataStatistics = pd.DataFrame(statisticGlobList).T
            dataStatistics.columns = [len(coeffs),len(coeffs)+1,len(coeffs)+2,len(coeffs)+3,len(coeffs)+4,len(coeffs)+5]
                
            # Gather information on energies
            dataEnergy = pd.DataFrame(energyList)

            for coeff in range(len(coeffs)):
                energyListGlob.append(np.amax(dataEnergy[coeff]))

            dataEnergy = pd.DataFrame(energyListGlob).T

            dataEnergyStat = dataEnergy.join(dataStatistics)
            
            dataFrequency = pd.DataFrame(frequencyList)

            for band in xrange(0, len(cutoff)-1):
                dataList.append(np.amax(dataFrequency[band]))

            dataFrequency = pd.DataFrame(dataList).T
            dataFrequency.columns = [len(coeffs)+6,len(coeffs)+7,len(coeffs)+8,len(coeffs)+9,len(coeffs)+10]

            data = dataEnergyStat.join(dataFrequency)

            # Write in HDFStore file the processed data
            if init == 0:
                hf.put('data', data, format='table')
                dataLenght = data.shape[0]
                print data.shape
            else:
                hf.append('data', data, format='table')
            
            if dataType == 'training':
                if init == 0:
                    hf.put('y', y, format='table')
                else:
                    hf.append('y', y, format='table')
            init = 1
        
        fileNumber += 1
        
        if fileNumber % 100 == 0.0:
            print '%d files done' % fileNumber
            
    hf.close()
    
    if dataType == 'training':  
        print 'Training done'
    else:
        print 'Testing done' 

    return fileNumber, dataLenght


if __name__=="__main__":
    #MatTrainingFiles= getFiles("train_1")
    #MatTestingFiles= getFiles("test_1")
    #NumberTrainingFiles, dataTrainingLenght = processData(MatTrainingFiles, 'training')
    #NumberTestingFiles, dataTestingLenght = processData(MatTestingFiles, 'testing')
    
    #computeFinalScore(MatTestingFiles)
    computeTestScore()
