
# coding: utf-8

# In[1]:

import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.neighbors
import sklearn.model_selection
from IPython import embed

# # Settings

# In[2]:

PATH_DATA_TRAIN = './train_data/'
FEATURE_MASK = [1,2,3,4,5,9]#[3, 4, 5]
PATH_MODEL = './model_final.pkl'


model_final = []

# # Load Train Data

# In[3]:

features_trn = []
targets_trn = []
filenames_trn = glob.glob(PATH_DATA_TRAIN + '*.xlsx')
for fname in filenames_trn:
    xls_file = pd.ExcelFile(fname)
    vect_fe = xls_file.parse('Sheet1')['Unnamed: 1'][1.0:10.0].as_matrix()
    features_trn.append(vect_fe)
    target = xls_file.parse('Sheet1')['Unnamed: 1'].as_matrix()[0]
    targets_trn.append(target)
    
features_trn = np.vstack(features_trn)[:,FEATURE_MASK]
targets_trn = np.vstack(targets_trn)


# In[4]:

#inds = (features_trn[:,2]!=-1)
#list(zip(features_trn[inds], targets_trn[inds]))
#plt.scatter(features_trn[:,-2] , targets_trn)
np.argsort(targets_trn.ravel())

mask = np.ones((features_trn.shape[0],), dtype=bool)
mask[[7,3,20,4]] = False
targets_trntrn = targets_trn[mask]
features_trn = features_trn[mask]
#plt.scatter(features_trn[:,-2] , targets_trn)


# In[ ]:




# In[5]:
scaler_mean = features_trn.mean(axis=0)
scaler_std = features_trn.astype(np.float).std(axis=0)
features_trntrn = (features_trn - scaler_mean) / scaler_std



# # Random Forest Regression

# In[6]:

def plot_regresssion(y_pred,y_target):
    return
    plt.ion()
    plt.figure()
    plt.plot(y_target,y_pred, 'x')
    plt.xlabel('Target')
    plt.ylabel('Estimate')

error_tstst = []
for ind_tstst in range(features_trn.shape[0]):
	rgr_models =  ()
	inds_trntrn = np.arange(0,features_trntrn.shape[0]) != ind_tstst
	features_trn = features_trntrn[inds_trntrn]
	targets_trn = targets_trntrn[inds_trntrn]
	# In[7]:

	targets_tst = []
	preds_tst = []
	accus_tst = []
	for ind_tst in range(features_trn.shape[0]):
	    inds_trn = np.arange(0,features_trn.shape[0]) != ind_tst
	    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=None,
		                                        max_features=None, bootstrap=True,
		                                        criterion='mae')
	    rgr.fit(features_trn[inds_trn],targets_trn[inds_trn].ravel())
	    pred_tst = rgr.predict(features_trn[ind_tst].reshape(1,-1))
	    preds_tst.append(pred_tst)
	    targets_tst.append(targets_trn[ind_tst])
	    error_tst = np.abs(pred_tst-targets_trn[ind_tst])
	    accus_tst.append(error_tst)
	    rgr_models += (rgr,)
	    
	print('Random Forest MAE: ' + str(np.mean(accus_tst)))
	plot_regresssion(preds_tst,targets_tst)


	# # Random Forest Regression (CV)

	# In[8]:

	targets_tst = []
	preds_tst = []
	accus_tst = []
	kf = sklearn.model_selection.KFold(n_splits=20)
	for inds_trn, inds_tst in kf.split(targets_trn):
	    #inds_trn = np.arange(0,features_trn.shape[0]) != ind_tst
	    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=None,
		                                        max_features=None, bootstrap=True,
		                                        criterion='mae')
	    rgr.fit(features_trn[inds_trn],targets_trn[inds_trn].ravel())
	    pred_tst = rgr.predict(features_trn[inds_tst])
	    preds_tst.append(pred_tst)
	    targets_tst.append(targets_trn[inds_tst])
	    error_tst = np.abs(pred_tst-targets_trn[inds_tst])
	    accus_tst.append(np.mean(error_tst))
	print('Random Forest MAE: ' + str(np.mean(accus_tst)))
	#plot_regresssion(preds_tst,targets_tst)


	# In[9]:

	rgr.feature_importances_


	# # Linear Regression

	# In[10]:

	targets_tst = []
	preds_tst = []
	accus_tst = []
	for ind_tst in range(features_trn.shape[0]):
	    inds_trn = np.arange(0,features_trn.shape[0]) != ind_tst
	    rgr = sklearn.linear_model.LinearRegression()
	    rgr.fit(features_trn[inds_trn],targets_trn[inds_trn].ravel())
	    pred_tst = rgr.predict(features_trn[ind_tst].reshape(1,-1))
	    targets_tst.append(targets_trn[ind_tst])
	    preds_tst.append(pred_tst)
	    error_tst = np.abs(pred_tst-targets_trn[ind_tst])
	    accus_tst.append(error_tst)
	    #print(rfe.support_)
	    rgr_models += (rgr,)
	    
	print('Linear Regression MAE: ' + str(np.mean(accus_tst)))
	plot_regresssion(preds_tst,targets_tst)


	# In[11]:

	rgr.coef_


	# # Linear Regression + Recursive Feature Elimination

	# In[12]:

	preds_tst = []
	accus_tst = []
	for ind_tst in range(features_trn.shape[0]):
	    inds_trn = np.arange(0,features_trn.shape[0]) != ind_tst
	    rgr = sklearn.linear_model.LinearRegression()
	    rfe = sklearn.feature_selection.RFECV(rgr, cv=10, n_jobs=-1)
	    rfe.fit(features_trn[inds_trn],targets_trn[inds_trn].ravel())
	    pred_tst = rfe.predict(features_trn[ind_tst].reshape(1,-1))
	    preds_tst.append(pred_tst)
	    error_tst = np.abs(pred_tst-targets_trn[ind_tst])
	    accus_tst.append(error_tst)
	    rgr_models += (rfe,)
	    #print(rfe.support_)
	print('Linear Regression and Recursive Feature Elimination MAE: ' + str(np.mean(accus_tst)))
	plot_regresssion(preds_tst,targets_tst)


	# # Ridge Regression 

	# In[13]:

	preds_tst = []
	accus_tst = []
	for ind_tst in range(features_trn.shape[0]):
	    inds_trn = np.arange(0,features_trn.shape[0]) != ind_tst
	    rgr = sklearn.linear_model.RidgeCV(alphas=2.0**np.arange(-20,20))
	    rgr.fit(features_trn[inds_trn],targets_trn[inds_trn].ravel())
	    pred_tst = rgr.predict(features_trn[ind_tst].reshape(1,-1))
	    preds_tst.append(pred_tst)
	    error_tst = np.abs(pred_tst-targets_trn[ind_tst])
	    accus_tst.append(error_tst)
	    rgr_models += (rgr,)
	    #print(rfe.support_)
	print('Ridge Regression MAE: ' + str(np.mean(accus_tst)))
	plot_regresssion(preds_tst,targets_tst)


	# # Ridge Regression + Recursive Feature Elimination

	# In[14]:

	preds_tst = []
	accus_tst = []
	for ind_tst in range(features_trn.shape[0]):
	    inds_trn = np.arange(0,features_trn.shape[0]) != ind_tst
	    rgr = sklearn.linear_model.RidgeCV(alphas=2.0**np.arange(-20,10))
	    rfe = sklearn.feature_selection.RFECV(rgr, cv=10, n_jobs=-1)
	    rfe.fit(features_trn[inds_trn],targets_trn[inds_trn].ravel())
	    pred_tst = rfe.predict(features_trn[ind_tst].reshape(1,-1))
	    preds_tst.append(pred_tst)
	    error_tst = np.abs(pred_tst-targets_trn[ind_tst])
	    accus_tst.append(error_tst)
	    rgr_models += (rfe,)
	    #print(rfe.support_)
	print('Ridge Regression and Recursive Feature Elimination MAE: ' + str(np.mean(accus_tst)))
	plot_regresssion(preds_tst,targets_tst)


	# In[15]:

	rfe.ranking_


	# # KNN Regression

	# In[16]:

	preds_tst = []
	accus_tst = []
	for ind_tst in range(features_trn.shape[0]):
	    inds_trn = np.arange(0,features_trn.shape[0]) != ind_tst
	    rgr = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
	    rgr.fit(features_trn[inds_trn],targets_trn[inds_trn].ravel())
	    pred_tst = rgr.predict(features_trn[ind_tst].reshape(1,-1))
	    preds_tst.append(pred_tst)
	    error_tst = np.abs(pred_tst-targets_trn[ind_tst])
	    accus_tst.append(error_tst)
	    rgr_models += (rgr,)
	    #print(rfe.support_)
	print('KNN MAE: ' + str(np.mean(accus_tst)))
	plot_regresssion(preds_tst,targets_tst)

	model_final += rgr_models
	# In[17]:
	pred_tstst = []
	for rr in rgr_models:
		pred_tstst.append(rr.predict(
			features_trntrn[ind_tstst].reshape(1,-1)))
		#pred_tstst.append(np.abs(rr.predict(
	        #	features_trntrn[ind_tstst].reshape(1,-1))-targets_trntrn[ind_tstst]))
	pred_tstst = np.mean(pred_tstst)
	error_tstst.append(np.abs(pred_tstst-targets_trntrn[ind_tstst]))

print(np.mean(error_tstst))

