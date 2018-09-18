import pickle
from functions import *

f = open('store.pckl', 'rb')
model_bank = pickle.load(f)
f.close()

pred_Y = []
for model in model_bank:
    pred_Y_part = model.predict_proba(test)[:,1]
    pred_Y.append(pred_Y_part)


pred = pd.DataFrame(pred_Y).T
pred.drop([3],axis = 1 , inplace= True)
pred['pred_Y'] = pred.mean(axis=1) 
pred_test = pred.pred_Y.values

fpr, tpr, thresholds = roc_curve(Y_test, pred_Y)
auc_result = auc(fpr, tpr)


    
'''
model_bank = []
for i in range(5):
    seed = 9 * (i + 1)
    model = training_xgb_model(X_train,Y_train,seed)
    model_bank.append(model)
    
f = open('store.pckl', 'wb')
pickle.dump(model_bank, f)
f.close()

'''  