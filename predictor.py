#!/usr/bin/python2.7

import re
import os
import shutil
import numpy as np
import pandas as pd
import scipy
from scipy.stats import entropy as ent
from scipy.io import savemat, loadmat
from functools import partial
from multiprocessing import Pool 
from sklearn.preprocessing import normalize
from scipy.signal import correlate, resample, coherence
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score
import h5py
from sklearn.grid_search import GridSearchCV
#import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
#import mne
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, freqz
#from joblib import Parallel, delayed

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

def createSubmission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('File,Class\n')
    total = 0
    for id in test['Id']:
        patient = id // 100000
        fid = id % 100000
        str1 = str(patient) + '_' + str(fid) + '.mat' + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

def ComputeScore(NumberFiles):
    X = None
    y = None
    
    # Read data from HDFStore file 
    with h5py.File('trainingDataT2.h5', 'r') as hf:
        #print 'List of arrays in this file: ', hf.keys()
        for i in range(NumberFiles):
            group = hf.get('file_%s' % i)
            #print 'list of items in the group: ', group.items()
            data = group.get('data')
            subY = group.get('y')
            if X is None:
                X = data
	    else:
                X = np.concatenate((X, data), axis=0)
            if y is None:
                y = subY
	    else:
                y = np.concatenate((y, subY), axis=0)
    '''
    for i in range(32):
        plt.subplot(16, 2, i + 1)
        plt.plot(X[i])
    plt.show()
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)[:, 1]
		
    result = roc_auc_score(y_test, predictions) 	

    return result
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

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

    kf = KFold(len(X_train), n_folds= 10, shuffle=True, random_state=42)

    scoring_fnc = make_scorer(f1_score)

    clf = GridSearchCV(clf, parameters, scoring_fnc, cv=kf)

    clf.fit(X_train, y_train)
    
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
            
    prediction = clf.predict(X_test)
            
    result = roc_auc_score(y_test, prediction)	

    return result
    '''
def processData(MatFiles, dataType):
        fileNumber = 0

        if dataType == 'training':  
            hf = h5py.File('trainingDataT2.h5', 'w')
        else:
            hf = h5py.File('testingDataT2.h5', 'w')
        
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
		sequence = datastruct['sequence'][0][0][0][0]
		channelIndices = datastruct['channelIndices'][0][0][0]
		iEEGsamplingRate = datastruct['iEEGsamplingRate'][0][0][0][0]
		nSamplesSegment = datastruct['nSamplesSegment'][0][0][0][0]
		ieegData = datastruct['data'][0][0].transpose()

		# extract the class of the file
		header = re.split(r"[_.]", os.path.basename (MatFile))
		category = header[2]

		subY = np.repeat(category, 24000, axis=0)
		subY = map(int, subY)

		freq = [0.5, 30]
                b,a = butter(5, np.array(freq)/(iEEGsamplingRate/2), btype='bandpass', analog=False)
                ieegData = np.array(signal.filtfilt(b ,a, ieegData[:16]))

                # Frequency vector
                f = iEEGsamplingRate*np.linspace(0,nSamplesSegment/10,nSamplesSegment/10)/nSamplesSegment                 

                for el in range(16):

                    Y = np.fft.fft(ieegData[el])

                    filtered = []
                    b= []               # store filter coefficient
                    cutoff = [0.5, 4.0, 7.0, 12.0, 30.0]

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
                    #ax = plt.subplot(16, 1, el+1)
                    #plt.plot(f,2*np.abs(Y[0:nSamplesSegment/10]))
                    for i in xrange(0, len(filtered)):
                        Y = filtered[i]
                        Y = np.fft.fft(Y[M/2:nSamplesSegment+M/2])
                        subData = abs(Y[0:nSamplesSegment/10])
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
                #plt.legend(['delta band, 0-4 Hz','theta band, 4-7 Hz','alpha band, 7-12 Hz','beta band, 12-30 Hz'])
                    
                #plt.show()         

                # Write in HDFStore file the processed data              
                entry = hf.create_group('file_%s' % fileNumber)
                entry.create_dataset('data', data=data)
                entry.create_dataset('y', data=subY)
                
		fileNumber += 1
		
        hf.close()
        
        if dataType == 'training':  
            print 'Training done'
        else:
            print 'Testing done' 

	return fileNumber


if __name__=="__main__":
	bestElectrodeIndices = []

	MatTrainingFiles= getFiles("train_2")
	MatTestingFiles= getFiles("test_2")
	NumberTrainingFiles = processData(MatTrainingFiles, 'training')
	NumberTestingFiles = processData(MatTestingFiles, 'testing')

	result = ComputeScore(NumberTrainingFiles)
	print "result : ", result

