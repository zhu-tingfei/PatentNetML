import os
import time
import random
import numpy as np
import pandas as pd
import pickle
import warnings
from math import *


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

import xgboost
from sklearn import metrics
from sklearn.metrics import *

from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

 
def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)



def undersampling(df_train,approach='random',num=1):
    patents = df_train.PATENT_ID.unique().tolist()
    df_train_undersampling = pd.DataFrame()
    df_train_rest = pd.DataFrame()
    for p in patents:
        df_p = df_train[df_train.PATENT_ID == p].reset_index(drop=True)
        # candidate and select negative sample
        n = df_p.shape[0]
        index_list = [*range(n)]
        candidate_index = [i for i in range(n) if df_p.Target[i] ==1][0]
        index_list.remove(candidate_index)
        get_index = [candidate_index]
        if approach == 'random':
            for n in range(num):
                random_index = random.choice(index_list)
                index_list.remove(random_index)
                get_index.append(random_index)
        elif approach == 'similar':
            X = df_p.drop(columns=['PATENT_ID','P_Ca_SMILES','Target'])
            X = X.values
            max_similarity = 0
            for index_i in index_list:
                smilarity = cosine_similarity(X[candidate_index],X[index_i])
                if smilarity > max_similarity:
                    max_similarity = smilarity
                    similar_index = index_i
                else:
                    pass
            get_index.append(similar_index)
        rest_index = index_list


        df_temp = df_p.iloc[get_index,]
        df_train_undersampling = pd.concat([df_train_undersampling,df_temp],ignore_index=True)

        df_temp = df_p.iloc[index_list,]
        df_train_rest = pd.concat([df_train_rest,df_temp],ignore_index=True)

    return df_train_undersampling,df_train_rest



def feature_selection(df_train,df_valid,df_test,wdirs='../',rounds=0):

    # Load the dataset
    X_train = df_train.drop(columns=['PATENT_ID','P_Ca_SMILES','Target'])
    X_valid = df_valid.drop(columns=['PATENT_ID','P_Ca_SMILES','Target'])
    X_test  = df_test.drop(columns=['PATENT_ID','P_Ca_SMILES','Target'])

    y_train = df_train.Target
    y_valid = df_valid.Target
    y_test  = df_test.Target

    # Save the column names
    column_names = X_test.columns.tolist()


    # Standardize the descriptors
    scaler = StandardScaler()
    X_std_train = scaler.fit_transform(X_train)
    X_std_valid = scaler.transform(X_valid)
    X_std_test  = scaler.transform(X_test)
    print('Standardize the descriptors:',len(column_names))
    
    # Eliminate features with small variance
    selector = VarianceThreshold(threshold=0.05)
    X_sel_train = selector.fit_transform(X_std_train)
    X_sel_valid = selector.transform(X_std_valid)
    X_sel_test  = selector.transform(X_std_test)
    column_names=[column_names[i] for i in selector.get_support(indices = True)]
    print('Eliminate features with small variance:',len(column_names))

    # Remove highly correlated features
    df_sel_train = pd.DataFrame(X_sel_train,columns=column_names)
    df_sel_valid = pd.DataFrame(X_sel_valid,columns=column_names)
    df_sel_test  = pd.DataFrame(X_sel_test,columns=column_names)
    
    corr_matrix = df_sel_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    
    X_corr_train = df_sel_train.drop(to_drop, axis=1)
    X_corr_valid = df_sel_valid.drop(to_drop, axis=1)
    X_corr_test  = df_sel_test.drop(to_drop, axis=1)

    column_names_=[i for i in column_names if i not in to_drop]
    print('Remove highly correlated:',len(column_names_))
    
    # Use the Boruta algorithm for feature selection
    rf = RandomForestClassifier(n_jobs=12, max_depth=3,random_state=123,
                                class_weight='balanced_subsample')
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=123)
    X_final_train = boruta_selector.fit_transform(np.array(X_corr_train), y_train)

    # Get the important features
    important_features = [column_names_[i] for i, val in enumerate(boruta_selector.support_) if val]
    print('Important features:',len(important_features))
    
    # Use the selected features for classification models
    X_final_train = X_corr_train[important_features]
    X_final_valid = X_corr_valid[important_features]
    X_final_test  = X_corr_test[important_features]

    print(important_features)
    # Save the key feature selection process
    fs_dict = {'fs1':scaler,
               'fs2':selector,
               'fs3':column_names,
               'fs4':important_features,
              }
    with open(f'{wdirs}/FeatureSelection_{rounds}.pkl', 'wb') as f:
        pickle.dump(fs_dict, f)
    
    return X_final_train,X_final_valid,X_final_test,y_train,y_valid,y_test




def main():

    warnings.filterwarnings("ignore")

    wdirs = f'../results/rf_all'#time.strftime("%Y-%m-%d-T%H-%M-%S", time.localtime())
    results_file = f'{wdirs}/results.csv'

    max_r = 0
    if os.path.exists('%s'%wdirs):
        file_exist = os.listdir(wdirs)
        for f in file_exist:
            if 'rf' in f:
                r = int(f.split('.')[0].split('_')[1])
                if r >max_r:
                    max_r = r
                else:
                    pass
    else:
        os.makedirs('%s'%wdirs)

        with open(results_file,'w+') as f:
            f.write(','.join(['rounds','criterion','min_samples_split','n_estimators','class_weight',
                              'train_auc','train_GA','train_BA','train_precision','train_recall','train_f1score','train_mcc',
                              'valid_auc','valid_GA','valid_BA','valid_precision','valid_recall','valid_f1score','valid_mcc',
                              'test_auc','test_GA','test_BA','test_precision','test_recall','test_f1score','test_mcc',
                              ]))
            f.write('\n')
    # load data
    df_train  = pd.read_csv('../data/mini-train.csv')   # Change to your own data path
    df_valid  = pd.read_csv('../data/mini-valid.csv')   # Change to your own data path
    df_test   = pd.read_csv('../data/mini-test.csv')    # Change to your own data path

    print('load data finished')


    rounds = 200

    for r in range(max_r,rounds):
        # undersampling
        negative_sample_num = 10
        df_train_undersampling,df_train_rest = undersampling(df_train,approach='random',num=negative_sample_num)
        df_train_undersampling = df_train_undersampling.sample(frac = 1)
        df_valid_add = pd.concat([df_valid,df_train_rest],ignore_index=True)
        df_valid_add = df_valid_add.sample(frac = 1)
        # feature selection
        X_train,X_valid,X_test,y_train,y_valid,y_test = feature_selection(df_train_undersampling,df_valid_add,df_test,wdirs,r)

        # release memory:
        df_train_undersampling = pd.DataFrame()
        df_train_rest = pd.DataFrame()
        df_valid_add = pd.DataFrame()

        # log file
        log_file = f'{wdirs}/parameters_{r}.csv'
        with open(log_file,'w+') as f:
            f.write(','.join(['criterion','min_samples_split','n_estimators','class_weight','valid_roc','test_roc'])+'\n')
        # model training


    # 要调的参数




        def f(criterion,min_samples_split,n_estimators,class_weight,direction = False):
            criterion_list = ['gini', 'entropy']
            class_weight_list = ['balanced','balanced_subsample']
            clf = RandomForestClassifier(criterion=criterion_list[int(round(criterion))],
                                        min_samples_split=int(round(min_samples_split)),
                                        n_estimators=int(round(n_estimators)),
                                        class_weight=class_weight_list[int(round(class_weight))],
                                        n_jobs=24)

            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_train)
            y_pred_proba = clf.predict_proba(X_train)[:, 1]
            train_auc = metrics.roc_auc_score(y_train, y_pred_proba)

            y_pred = clf.predict(X_valid)
            y_pred_proba = clf.predict_proba(X_valid)[:, 1]
            valid_auc = metrics.roc_auc_score(y_valid, y_pred_proba)


            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            test_auc = metrics.roc_auc_score(y_test, y_pred_proba)


            with open(log_file,'a') as f:
                f.write(','.join([str(int(round(criterion))),str(int(round(min_samples_split))),str(int(round(n_estimators))),str(int(round(class_weight)))]))
                f.write(','+str(valid_auc)+','+str(test_auc)+'\n')

            # GPGO maximize performance by default, set performance to its negative value for minimization
            if direction:
                return -valid_auc
            else:
                return valid_auc


    ### Bayesian optimization

        cov = matern32()
        gp = GaussianProcess(cov)
        acq = Acquisition(mode='UCB')
        param = {
                 'criterion': ('int', [0, 1]),
                 'min_samples_split':('int', [2, 10]),
                 'n_estimators': ('int', [50, 1000]),
                 'class_weight': ('int', [0, 1]),
                 }
        np.random.seed(168)
        gpgo = GPGO(gp, acq, f,param,n_jobs=24)
        gpgo.run(max_iter=50,init_evals=3)
        results = gpgo.getResult()
        print(results)
        best_param = results[0]
        best_result = results[1]
        # best
        best_criterion = best_param.get('criterion')
        best_n_estimators = best_param.get('n_estimators')
        best_min_samples_split = best_param.get('min_samples_split')
        best_class_weight = best_param.get('class_weight')

        criterion_list = ['gini', 'entropy']
        class_weight_list = ['balanced','balanced_subsample']
        clf = RandomForestClassifier(criterion=criterion_list[best_criterion],
                                    min_samples_split=int(round(best_min_samples_split)),
                                    n_estimators=int(round(best_n_estimators)),
                                    class_weight=class_weight_list[best_class_weight])

        clf.fit(X_train, y_train)

        with open(f'{wdirs}/rf_{r}.pkl', 'wb') as f:
            pickle.dump(clf, f)

        y_pred = clf.predict(X_train)
        y_pred_proba = clf.predict_proba(X_train)[:, 1]
        
        train_mcc = metrics.matthews_corrcoef(y_train, y_pred)
        train_precision = metrics.precision_score(y_train, y_pred)
        train_recall = metrics.recall_score(y_train, y_pred)
        train_f1score = metrics.f1_score(y_train, y_pred)
        train_auc = metrics.roc_auc_score(y_train, y_pred_proba)
        train_GA = metrics.accuracy_score(y_train, y_pred)
        train_BA = metrics.balanced_accuracy_score(y_train, y_pred)

        y_pred = clf.predict(X_valid)
        y_pred_proba = clf.predict_proba(X_valid)[:, 1]

        valid_mcc = metrics.matthews_corrcoef(y_valid, y_pred)
        valid_precision = metrics.precision_score(y_valid, y_pred)
        valid_recall = metrics.recall_score(y_valid, y_pred)
        valid_f1score = metrics.f1_score(y_valid, y_pred)
        valid_auc = metrics.roc_auc_score(y_valid, y_pred_proba)
        valid_GA = metrics.accuracy_score(y_valid, y_pred)
        valid_BA = metrics.balanced_accuracy_score(y_valid, y_pred)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        test_mcc = metrics.matthews_corrcoef(y_test, y_pred)
        test_precision = metrics.precision_score(y_test, y_pred)
        test_recall = metrics.recall_score(y_test, y_pred)
        test_f1score = metrics.f1_score(y_test, y_pred)
        test_auc = metrics.roc_auc_score(y_test, y_pred_proba)
        test_GA = metrics.accuracy_score(y_test, y_pred)
        test_BA = metrics.balanced_accuracy_score(y_test, y_pred)



        with open(results_file,'a') as f:
            f.write(','.join([str(r),str(best_criterion),str(best_min_samples_split),str(best_n_estimators),str(best_class_weight),
                              str(train_auc),str(train_GA),str(train_BA),str(train_precision),str(train_recall),str(train_f1score),str(train_mcc),
                              str(valid_auc),str(valid_GA),str(valid_BA),str(valid_precision),str(valid_recall),str(valid_f1score),str(valid_mcc),
                              str(test_auc), str(test_GA), str(test_BA), str(test_precision), str(test_recall), str(test_f1score), str(test_mcc)]))
            f.write('\n')


if __name__ == "__main__":
    main()




