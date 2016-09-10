#!/usr/bin/python2.7

import re
import os
import shutil
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
from functools import partial
from multiprocessing import Pool 
from sklearn.preprocessing import normalize
from scipy.signal import correlate, resample
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score
import h5py
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

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

def verifyElectrodeValue(X, y):
	electrode = 0
	bestElectrodeIndices = []

	clf = RandomForestClassifier(n_estimators = 100)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	for electrode in range(16):	
		data = []
		globalY = []
		i = electrode
		while i < len(X_train):
			data.append(X_train[i].transpose())
			globalY.append(y_train[i])
			i += 16

		clf.fit(data, globalY)
		predictions = clf.predict_proba(X_test)[:, 1]
		
		result = roc_auc_score(y_test, predictions) 

		if result > 0.3:
			bestElectrodeIndices.append(electrode)
                        print result
	
	print bestElectrodeIndices	

	return bestElectrodeIndices 

def ComputeGlobalScore(X, y, electrodesIndices):
	data=[]
	globalY = []
	clf = RandomForestClassifier(n_estimators = 100)

	for electrode in electrodesIndices:
		while electrode < len(X):
			data.append(X[electrode].transpose())
			globalY.append(y[electrode])
			electrode += 16		

	X_train, X_test, y_train, y_test = train_test_split(data, globalY, test_size=0.33, random_state=42)

	clf.fit(X_train, y_train)
	predictions = clf.predict_proba(X_test)[:, 1]
		
	result = roc_auc_score(y_test, predictions) 	

	return result

def ComputeScore(NumberFiles):
    X = None
    y = None
    
    # Read data from HDFStore file 
    with h5py.File('trainingData.h5', 'r') as hf:
        print 'List of arrays in this file: ', hf.keys()
        for i in range(NumberFiles):
            group = hf.get('file_%s' % i)
            print 'list of items in the group: ', group.items()
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
        'eta': [0.01, 0.015, 0.025, 0.05, 0.1],
        'gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'lambda': [0.05, 0.1, 1.0],
        'alpha': [0, 0.1, 0.5, 1.0],
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

def processData(MatFiles):
        data = None
	y = None
        fileNumber = 0
        
        hf = h5py.File('trainingData.h5', 'w')
        
	# Obtain data
	for MatFile in MatFiles:
		try:
			d = loadmat(MatFile)
		except ValueError as ex:
			print(u'Error loading MAT file {}: {}'.format(os.path.basename (MatFile), str(ex)))

		# Extract each data from the file
		datastruct = d['dataStruct']
		sequence = datastruct['sequence'][0][0][0][0]
		channelIndices = datastruct['channelIndices'][0][0][0]
		iEEGsamplingRate = datastruct['iEEGsamplingRate'][0][0][0][0]
		nSamplesSegment = datastruct['nSamplesSegment'][0][0][0][0]
		ieegData = datastruct['data'][0][0].transpose()
		
                # Remove measurement errors
                for i in range(ieegData.shape[0]):
                    for j in range(ieegData.shape[1]):
                        if ieegData[i][j] == 0.0:
                            for k in range(ieegData.shape[0]):
                                if ieegData[k][j] != 0.0:
                                    noError = 1
                            if noError != 1:
                                print "----------Measurement error----------"
                                for k in range(ieegData.shape[0]):
                                    ieegData[k][j] = ieegData[k][j-1]                           
                        #if np.abs(ieegData[i][j]-ieegData[i][j-1]) > 70.0:
                        #    ieegData[i][j] = ieegData[i][j-1]

		# extract the class of the file
		header = re.split(r"[_.]", os.path.basename (MatFile))
		category = header[2]

		subY = np.repeat(category, 16, axis=0)
		subY = map(int, subY)

		# Compute the FFT and normalize the result
		ieegData = np.fft.fft(ieegData)
		ieegData = normalize(ieegData, copy=False)

                # Write in HDFStore file the processed data              
                entry = hf.create_group('file_%s' % fileNumber)
                entry.create_dataset('data', data=ieegData)
                entry.create_dataset('y', data=subY)               

		fileNumber += 1
		
        hf.close()
        
	#print 'y_shape = ', y
	#print 'X_shape = ', data.shape

	# Convert dictionnary of data to a dataFrame
	#new_dataFrame = pd.DataFrame(dataDict.items(), columns= ['DataNumber','Data'])

	return fileNumber


if __name__=="__main__":
	bestElectrodeIndices = []

	MatFiles= getFiles("init_1")
	NumberFiles = processData(MatFiles)
        
	#bestElectrodeIndices = verifyElectrodeValue(X, y)

	#result = ComputeGlobalScore(X, y, bestElectrodeIndices)
	#print "result with electrodes sorted : ", result

	#result = ComputeScore(2)
	#print "result with electrodes not sorted : ", result

