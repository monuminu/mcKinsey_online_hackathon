'''
------------------------------------------Importing the Libraries--------------------------------------------
'''
print('Importing the Libraries Started')
import numpy as np
import pandas as pd
from swifter import swiftapply
import os
from functions import *
from sklearn.model_selection import train_test_split
import math
from imblearn.over_sampling import SMOTE 
import pickle
def main():
    '''
    -----------------------------------------Setting the Home Directory--------------------------------------------
    '''
    print('Setting the Home Directory Started')
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    '''
    ------------------------------------------Importing the Dataseet--------------------------------------------
    '''
    print('Importing the Dataseet Started')
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    test_bkp = test
    data  = pd.concat([train,test],axis = 0).reset_index()
    pd.options.display.float_format = '{:.4f}'.format 
    '''
    ----------------------------------------Checking for data details--------------------------------------------
    '''
    print('Checking for data details Started')
    #print(data.isnull().sum())  #To Check missing values per column
    #print(data.describe())      #To Check the Descriptive Statistics of the Data
    #print(data.dtypes)          #To Check the Data type of each column
    
    
    '''
    ----------------------------------------Missing values Treatment--------------------------------------------
    '''
    print('Missing values Treatment Started')
    data['Count_3-6_months_late'] = data['Count_3-6_months_late'].fillna(0)
    data['Count_6-12_months_late'] = data['Count_6-12_months_late'].fillna(0)
    data['Count_more_than_12_months_late'] = data['Count_more_than_12_months_late'].fillna(0)
    #data['application_underwriting_score'] = data['application_underwriting_score'].fillna(data['application_underwriting_score'].dropna().mean())
    
    
    '''
    ----------------------------------------Feature Transformation--------------------------------------------
    '''
    print('Feature Transformation Started')
    data['age_in_years'] = round(data['age_in_days'] / 365 ,2)
    data['age_category'] = swiftapply(data.age_in_years,get_age_category)
    data['income_category'] = swiftapply(data.Income,get_income_category)
    data['Total_Count'] = data['Count_3-6_months_late'] + data['Count_6-12_months_late'] + data['Count_more_than_12_months_late']
    data['if_ever_delayed'] = swiftapply(data['Total_Count'],get_ever_delayed)
    data.loc[data.application_underwriting_score.isnull(),'application_underwriting_score'] = data.groupby('age_category').application_underwriting_score.transform('mean')
    data = data[data.Income < 25000000]
    #data['Income'] = data["Income"].apply(np.log)
    data['ability_to_pay'] = data['premium'] / data['Income']
    data['Income_sqr'] = data['Income'] * data['Income']
    data['Income_cube'] = data['Income'] * data['Income'] * data['Income']
    data['payment_type'] = swiftapply(data['perc_premium_paid_by_cash_credit'],payment_type)
    data['delay_to_paid_ratio'] =  data['Total_Count']/data['no_of_premiums_paid']
    data['sourcing_channel'] = swiftapply(data['sourcing_channel'],get_source_channel)
    '''
    ----------------------------------------Feature Engineering--------------------------------------------
    '''
    print('Feature Engineering Started')
    data = drop_cols_with_unique_values(data)
    col_to_scale = ['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','Income',
                    'application_underwriting_score','no_of_premiums_paid','perc_premium_paid_by_cash_credit',
                    'premium','age_in_years','Total_Count','ability_to_pay','Income_sqr','Income_cube','age_in_days',
                    'delay_to_paid_ratio','sourcing_channel']
    
    for col in col_to_scale:
        data[col] = scaling(data,col)
    #One Hot Encoding the data
    
    #data = one_hot(data , 'sourcing_channel')   
    data = one_hot(data , 'residence_area_type')   
    data = one_hot(data , 'age_category')   
    data = one_hot(data , 'income_category') 
    data = one_hot(data , 'if_ever_delayed') 
    data = one_hot(data , 'payment_type') 
    
    data.drop(['id','age_in_years','index'],axis = 1 , inplace= True)  #Dropping in Age In Days Since we are going with Age in Years
    
    '''
    -------------------------------------Build Machine Learning Model--------------------------------------------
    '''
    print('Splitting Train and Test')
    print(data.head(5))
    #Spliting the train , test after the preprocessing the result [test date will go for Final evaluation in Hackathon]
    test = data[data.renewal.isnull()]
    train = data[~data.renewal.isnull()]
    
    X_train, X_test, Y_train, Y_test = train_test_split(train.loc[:,train.columns!='renewal']
                                                        ,train.renewal, test_size=0.2, random_state=35)
    
    seed = 27
    cols = X_train.columns
    sm = SMOTE(random_state=42)
    X_train, Y_train = sm.fit_sample(X_train, Y_train)
    X_train = pd.DataFrame(X_train,columns=cols)
    #trained with LGBM
    print('Model Training Started')
    model1 = training_xgb_model(X_train,Y_train,seed)
    #model2 = training_lgbm_model(X_train,Y_train)
    evaluate_xgb(model1,X_test,Y_test)
    #evaluate(model2,X_test,Y_test) 
    
    
    print('Predicting Test Data and Submission')
    #Submit the prediction of Original test data in the hackathon website 
    test = test.drop(['renewal'],axis = 1,inplace = False)
    pred_test1 = model1.predict_proba(test)[:,1]
    #pred_test2 = model2.predict(test)
    pred_test = pred_test1 #(pred_test1 + pred_test2)/2
    submission = pd.DataFrame({'id': test_bkp.id, 'premium': test_bkp.premium, 'renewal': pred_test})
    submission['incentives'] = 400*(np.log(submission['premium']) - np.log(10) - 2)
    submission['incentives'] = np.minimum(submission['incentives'],((1-submission['renewal'])*submission['premium']))
    submission.drop(['premium'],axis = 1 , inplace= True)
    submission.to_csv('submission7.csv',index=False)
    print(submission.head(5))   

if __name__ == '__main__':
    main()
