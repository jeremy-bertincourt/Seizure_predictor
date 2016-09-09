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
#import xgboost as xgb
        

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

def ComputeScore(X, y):

	clf = RandomForestClassifier(n_estimators = 100)	

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	clf.fit(X_train, y_train)
	predictions = clf.predict_proba(X_test)[:, 1]
		
	result = roc_auc_score(y_test, predictions) 	

	return result

def processData(MatFiles):
	data = None
	y = None

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
		
		if y is None:
			y = subY
		else:
			y = np.concatenate((y, subY), axis=0)
		if data is None:
			data = ieegData
		else:
			data = np.concatenate((data, ieegData), axis=0)

	#print 'y_shape = ', y
	#print 'X_shape = ', data.shape

	# Convert dictionnary of data to a dataFrame
	#new_dataFrame = pd.DataFrame(dataDict.items(), columns= ['DataNumber','Data'])

	# return array of data and class 
	#return new_dataFrame['Data'], y

	return data, y


if __name__=="__main__":
	bestElectrodeIndices = []

	MatFiles= getFiles("init_test")
	X, y = processData(MatFiles)

        # Write in HDFStore file the processed data 
        with h5py.File('data.h5', 'w') as hf:
            hf.create_dataset('data', data=X)
            hf.create_dataset('y', data=y)
            hf.close()

	bestElectrodeIndices = verifyElectrodeValue(X, y)

	result = ComputeGlobalScore(X, y, bestElectrodeIndices)
	#print "result with electrodes sorted : ", result

	result = ComputeScore(X, y)
	print "result with electrodes not sorted : ", result



