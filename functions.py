'''
------------------------------------------Importing the Libraries--------------------------------------------
'''
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import lightgbm as lgb
import numpy as np
import math
from sklearn.metrics import roc_curve, auc, accuracy_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
'''
-----------------------------------------Function Definations--------------------------------------------
'''
def get_ever_delayed(Total_Count):
    if Total_Count > 0:
        return 'Yes'
    else : return 'No'
    
def payment_type(prcnt_paid_cash):
    if prcnt_paid_cash == 0:
        return 'Full_Online'
    elif prcnt_paid_cash > 0 and prcnt_paid_cash< 1:
        return 'Partial'
    else : return 'Full_Cash'
    
def get_age_category(age):
    if age <= 18 :
        return 'Child'
    elif age > 18 and age <= 30 :
        return 'Young_Adult'
    elif age > 30 and age <= 45 :
        return 'Adult'
    elif age > 45 and age <= 60:
        return 'Middle_Age'
    else : return 'Old_Age'

def get_income_category(income):
    if income <= 107560:
        return 'Poor'
    elif income > 107560 and income <= 165240:
        return 'Lower_Middle'
    elif income > 165240 and income <= 250450:
        return 'Upper_Middle'
    elif income > 250450 and income <= 20000000:
        return 'Rich'    
    else : return 'Super_Rich'

def drop_cols_with_unique_values(df):
    #check for columns having only a single unique values 
    col_with_single_unique_value = []
    for col in df.columns:
        if len(df[col].unique()) == 1 :
            col_with_single_unique_value.append(col)

    df.drop(col_with_single_unique_value,axis = 1 , inplace= True)
    return df

def scaling(data,colname):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data[[colname]])

def get_source_channel(s):
    if s == 'A':
        return 0
    elif s =='B':
        return 1
    elif s =='C':
        return 2
    elif s =='D':
        return 3
    else : return 4  
    

def one_hot(data,colname):
    one_hot_encoded_data = pd.get_dummies(data[[colname]],prefix = colname,drop_first= True)
    data.drop([colname],axis = 1,inplace = True)
    data = pd.concat([data,one_hot_encoded_data],axis = 1)
    return data

def evaluate(model, test_features, test_labels):
    pred_Y = model.predict(test_features)
    fpr, tpr, thresholds = roc_curve(test_labels, pred_Y)
    auc_result = auc(fpr, tpr)
    print(auc_result)
    
def predict_model_bank(model_bank , test_features,test_labels):
    return [model.predict_proba(test_features)[:,1] for model in model_bank]
    
def evaluate_xgb(model, test_features, test_labels):
    pred_Y = model.predict_proba(test_features)[:,1]
    fpr, tpr, thresholds = roc_curve(test_labels, pred_Y)
    auc_result = auc(fpr, tpr)
    print(auc_result)
    
def training_lgbm_model(train_X,train_Y):
    lgb_params = {
        'objective' : 'binary', 
        'n_jobs' : 4, 
        'is_unbalance' :True, 
        'num_threads' :4, 
        'two_round' :True,
        'bagging_fraction' :0.9,
        'bagging_freq' :1,
        'boosting_type' : 'gbdt',
        'feature_fraction' : 0.9,
        'learning_rate' : 0.05,
        'metric':['auc'],
        'min_child_samples' : 10,
        'min_child_weight' : 5,
        'min_data_in_leaf' : 20,
        'min_split_gain' : 0.0,
        'n_estimators' : 1500,
        'num_leaves' : 300,
        'reg_alpha' : 0.0,
        'reg_lambda' : 0.0,
        'subsample' : 1.0
         }    
    # form LightGBM datasets
    dtrain_lgb = lgb.Dataset(train_X, label=train_Y)
    # LightGBM, cross-validation
    cv_result_lgb = lgb.cv(lgb_params, 
                           dtrain_lgb, 
                           num_boost_round=3000, 
                           nfold=5, 
                           stratified=True, 
                           early_stopping_rounds=50, 
                           seed= 2,
                           show_stdv=True)
    num_boost_rounds_lgb = len(cv_result_lgb['auc-mean'])
    print('Mean AUC : ' , np.mean(cv_result_lgb['auc-mean']))
    print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))
    # train model
    model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=num_boost_rounds_lgb)
    return model_lgb




def training_xgb_model2(train_X,train_Y,seed=27):
    xgb_model = xgb.XGBClassifier()
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [0.005,0.01,0.05], #so called `eta` value
                  'max_depth': [1,5,10,20],
                  'min_child_weight': [1,5,10,20],
                  'silent': [1],
                  'subsample': [0.8,0.6,0.2],
                  'colsample_bytree': [0.7,0.5,0.3],
                  'n_estimators': [5,50,500], #number of trees, change it to 1000 for better results
                  'missing':[-999],
                  'seed': [1337]}
    
    
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                       cv=StratifiedKFold(train_Y, n_folds=5, shuffle=True), 
                       scoring='roc_auc',
                       verbose=2, refit=True)
    
    clf.fit(train_X, train_Y)
    
    #trust your CV!
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))   
    return clf    
        
def training_xgb_model(train_X,train_Y,seed=27):
    
    xgb2 = XGBClassifier(
     learning_rate =0.01,
     n_estimators=1600,
     max_depth=6,
     min_child_weight=11,
     gamma=0,
     subsample=0.6,
     colsample_bytree=0.6,
     objective= 'binary:logistic',
     nthread=4,
     n_jobs= 4,
     reg_alpha = 0.01,
     scale_pos_weight=2,
     seed=seed)
    model = modelfit(xgb2, train_X, train_Y)
    return model

def modelfit(alg, train_X, train_Y,useTrainCV=True, cv_folds=5, early_stopping_rounds=30):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_X.values, label=train_Y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train_X, train_Y,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(train_X)
    dtrain_predprob = alg.predict_proba(train_X)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_Y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_Y, dtrain_predprob))
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')    
    return alg
    
def fetch_incentive(premium):
    premium = float(premium)
    print(premium)
    return 400*(math.log(premium) - math.log(10) - 2)
    
def d_fun_x(x):
    return (math.exp(2*(math.exp(-1*x/400))-(x/400)-2))/10