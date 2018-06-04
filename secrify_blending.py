from collections import Counter
import os
import random
import itertools

import operator

import pickle

import pandas as pd
import matplotlib.pyplot as plt

#SciPy
import scipy.stats as st
import scipy

#ML
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

import numpy
import numpy as np

import copy
from scipy.stats import randint
from scipy.stats import uniform
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
#from keras.layers import Dropout

three_to_one = {
        "ala" : "A",
        "arg" : "R",
        "asn" : "N",
        "asp" : "D",
        "cys" : "C",
        "glu" : "E",
        "gln" : "Q",
        "gly" : "G",
        "his" : "H",
        "ile" : "I",
        "leu" : "L",
        "lys" : "K",
        "met" : "M",
        "phe" : "F",
        "pro" : "P",
        "ser" : "S",
        "thr" : "T",
        "trp" : "W",
        "tyr" : "Y",
        "val" : "V"}

def count_substring(string,sub_string):
    l=len(sub_string)
    count=0
    for i in range(len(string)-len(sub_string)+1):
        if(string[i:i+len(sub_string)] == sub_string ):      
            count+=1
    return count 
        
def split_seq(a, n):
    k, m = divmod(len(a), n)
    return(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_all_poss_aa(seqs):
    poss_aa = set()
    for seq in seqs:
        for aa in Counter(seq).keys():
            poss_aa.add(aa)
    return(list(poss_aa))

def count_aa(seq,aa_order=[]):
    feat_vector = []
    counted_aa = Counter(seq)
    for aa in aa_order:
        try: feat_vector.append(counted_aa[aa])
        except: feat_vector.append(0)
    return(feat_vector)

def get_count_aa(seq,aa_to_count):
    total = 0
    for aa in aa_to_count:
        total += seq.count(aa)
    return(total)

def apply_prop_lib(seq,lib,avg=True):
    ret_val = 0.0
    for aa in seq:
        ret_val += lib[aa]
    if avg: 
        try: ret_val = ret_val/float(len(seq))
        except ZeroDivisionError: ret_val = 0.0
    return(ret_val)

def analyze_lib(infile_name):
    infile = open(infile_name)
    prop_dict = {}
    for line in infile:
        line = line.strip()
        if len(line) == 0: continue
        aa_three,val = line.lower().split(": ")
        val = float(val)
        prop_dict[three_to_one[aa_three]] = val
    return(prop_dict)

def get_set_all_aa(infile_name,ret_duplicated_pos=False):
    infile = open(infile_name)
    if ret_duplicated_pos: locs = []
    else: locs = set()
    gene_ident_to_row_ident = {}

    for line in infile:
        if line.startswith("identifier"): continue

        splitline = line.split(",")
        ident = splitline[1]
        start_pos = int(splitline[7])
        end_pos = int(splitline[8])
        
        if ident in gene_ident_to_row_ident.keys():
            gene_ident_to_row_ident[ident].append(splitline[0])
        else:
            gene_ident_to_row_ident[ident] = [splitline[0]]
        
        for pos in range(start_pos,end_pos+1):
            if ret_duplicated_pos: locs.append("%s|%s" % (ident,str(pos)))
            else: locs.add("%s|%s" % (ident,str(pos)))
    
    return(locs,gene_ident_to_row_ident)

def get_set_all_aa_compare(infile_name,locs_other,return_agg_seq=True,min_overlap_agg=0.95):
    infile = open(infile_name)
    num_in_other = 0
    num_total = 0
    distrib_overlap = []
    agg_seqs = []
    
    for line in infile:
        if line.startswith("identifier"): continue
        
        splitline = line.split(",")
        ident = splitline[1]
        seq = splitline[10]
        start = int(splitline[7])
        end = int(splitline[8])
        
        in_depleted = False
        tot_overlap = 0
        agg_seq = ""
        indexes_aa = []

        for index,pos in enumerate(range(start,end)):
            index_aa = int((index - (index % 3))/3)
            ident_loc = "%s|%s" % (ident,str(pos))
            if ident_loc in locs_other:
                in_depleted = True
                tot_overlap += 1
            else:
                if index_aa in indexes_aa: continue
                indexes_aa.append(index_aa)
                agg_seq += seq[index_aa]
        if tot_overlap/float(len(range(start,end+1))) > min_overlap_agg:
            agg_seqs.append(list(zip(agg_seq,indexes_aa)))
        distrib_overlap.append(tot_overlap/float(len(range(start,end+1))))
        num_total += 1
        if in_depleted:    num_in_other += 1
    return(distrib_overlap,agg_seqs,num_total,num_in_other)
    
def get_libs_aa(dirname):
    path = dirname
    listing = os.listdir(path)
    libs_prop = {}
    for infile in listing:
        if not infile.endswith(".txt"): continue
        libs_prop["".join(infile.split(".")[:-1])] = analyze_lib(os.path.join(path,infile))
    return(libs_prop)

def get_all_seqs(infile_name):
    infile = open(infile_name)
    seqs = []
    outfasta = open("all_seqs.mfa","w")
    for line in infile:
        if line.startswith("identifier"): continue
        
        line = line.strip()
        splitline = line.split(",")
        seq = splitline[10]
        seqs.append(seq)
        
        outfasta.write(">%s\n" % (splitline[0]))
        outfasta.write("%s\n" % (seq))
    outfasta.close()
    return(seqs)

def get_feats_simple_seq(seq,libs_prop):
    new_instance = {}
    for name,lib in libs_prop.items():
        new_instance[name] = apply_prop_lib(seq,lib)
    return(new_instance)

def get_feats_simple(infile_name,libs_prop,assign_class=1,nmer_feature="nmer_features.txt"):
    infile = open(infile_name)
    nmers = open(nmer_feature).readlines()

    data_df = []
    rownames = []

    for line in infile:
        if line.startswith("identifier"): continue
        
        line = line.strip()
        splitline = line.split(",")
        seq = splitline[10]
        rownames.append(splitline[0])
        #print(splitline[1])
        new_instance = {}
        new_instance["Ensembl_geneID"] = splitline[1]
        for name,lib in libs_prop.items():
            new_instance[name] = apply_prop_lib(seq,lib)
        #if not assign_class == False:
        new_instance["class"] = assign_class
        if splitline[0].startswith("Sc"): new_instance["organism"] = 1
        else: new_instance["organism"] = 0
        #new_instance["logFC_rep5"] = float(splitline[-1])
        #new_instance["logFC_rep4"] = float(splitline[-2])
        #new_instance["logFC_rep3"] = float(splitline[-3])
        
        for nmer in nmers:
            new_instance["countnmer|"+nmer] = count_substring(seq,nmer)
        
        data_df.append(new_instance)
    return(data_df,rownames)

def train_rnn(X,y,aa_order):
    translate_to_numeric = dict(zip(aa_order,range(1,len(aa_order)+1)))
    X_encoded = []
    for seq in X:
        instance = []
        for aa in seq:
            instance.append(translate_to_numeric[aa])
        X_encoded.append(instance)

    X_encoded = sequence.pad_sequences(X_encoded, maxlen=250)

    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(len(aa_order), embedding_vecor_length, input_length=250))
    model.add(Dropout(0.5))
    model.add(LSTM(250,return_sequences=True))
    #model.add(Dropout(0.5))
    model.add(LSTM(250,return_sequences=True))
    #model.add(Dropout(0.5))
    model.add(LSTM(250))
    model.add(Dropout(0.5))
    #model.add(Dense(100, activation='sigmoid'))
    #model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())

    model.fit(X_encoded, y, nb_epoch=100, batch_size=64, shuffle=True)

def train_svc(X,y,mod_number=1):
    svc_handle = SVC()

    param_dist = {'C': uniform(0.0,40), #,1000,3000], #
                  'kernel': ["rbf"], #,"poly","sigmoid","rbf",
                  'max_iter' : [100],
                  'probability' : [True]
    }

    n_iter_search = 3
    random_search = RandomizedSearchCV(svc_handle, param_distributions=param_dist,
                                       n_iter=n_iter_search,verbose=10,scoring="roc_auc",
                                       n_jobs=4,cv=StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

    random_search = random_search.fit(X, y)
    
    rbfsvc_model = random_search.best_estimator_
    
    random_search.feats = X.columns
    
    print(random_search.best_params_)
    print(random_search.best_score_)
    
    xgb_model = random_search.best_estimator_
    pickle.dump(random_search,open("svc_mod%s.pickle" % (mod_number),"wb"))

def train_ada(X,y,mod_number=1,cv=None):
    ada_handle = AdaBoostClassifier()

    param_dist = {'n_estimators': randint(10,100), #,1000,3000], #
                  'learning_rate': uniform(0.01,0.5)
    }

    n_iter_search = 10
    random_search = RandomizedSearchCV(ada_handle, param_distributions=param_dist,
                                       n_iter=n_iter_search,verbose=10,scoring="roc_auc",
                                       n_jobs=8,cv=cv)

    random_search = random_search.fit(X, y)

    ada_model = random_search.best_estimator_
    
    print(random_search.best_params_)
    print(random_search.best_score_)
    random_search.feats = X.columns
    ada_model = random_search.best_estimator_
    pickle.dump(random_search,open("ada_mod%s.pickle" % (mod_number),"wb"))

def train_ada_cv(X,y,mod_number=1,cv=None,outfile="mod.pickle"):
    ada_handle = AdaBoostClassifier()

    param_dist = {'n_estimators': randint(10,100), #,1000,3000], #
                  'learning_rate': uniform(0.01,0.5)
    }

    n_iter_search = 5
    
    test_preds = dict()
    for train_set,test_set in cv:
        X_temp = X.ix[train_set,:]
        y_temp = y.ix[train_set]
        X_test = X.ix[test_set,:]
        #print(X_temp)
        #,test_set)
        random_search = RandomizedSearchCV(ada_handle, param_distributions=param_dist,
                                           n_iter=n_iter_search,verbose=10,scoring="roc_auc", #["accuracy",]
                                           n_jobs=7,refit=True,cv=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

        random_search_res_xgb = random_search.fit(X_temp, y_temp)
        
        test_preds.update(dict(zip(test_set,random_search_res_xgb.predict_proba(X_test)[:,1])))
    
    random_search = RandomizedSearchCV(ada_handle, param_distributions=param_dist,
                                           n_iter=n_iter_search,verbose=10,scoring="roc_auc", #["accuracy",]
                                           n_jobs=7,refit=True,cv=cv) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

    random_search_res_xgb = random_search.fit(X, y)
    
    random_search.feats = X.columns
    ada_model = random_search.best_estimator_
    pickle.dump(random_search,open(outfile,"wb"))
        
    return(test_preds)

def train_rf(X,y,mod_number=1,cv=None):
    svc_handle = RandomForestClassifier()

    param_dist = {"n_estimators"      : randint(5,25),
                  "max_depth"         : randint(1,30),
                  "min_samples_split" : randint(100, 5000),
                  "min_samples_leaf"  : randint(100, 5000),
                  "bootstrap"         : [True, False]}

    n_iter_search = 3
    random_search = RandomizedSearchCV(svc_handle, param_distributions=param_dist,
                                       n_iter=n_iter_search,verbose=10,scoring="roc_auc",
                                       n_jobs=8,cv=cv)

    random_search = random_search.fit(X, y)

    rf_model = random_search.best_estimator_
    
    print(random_search.best_params_)
    print(random_search.best_score_)
    
    random_search.feats = X.columns
    rf_model = random_search.best_estimator_
    pickle.dump(random_search,open("rf_mod%s.pickle" % (mod_number),"wb"))
    
def train_rf_cv(X,y,mod_number=1,cv=None,outfile="mod.pickle"):
    svc_handle = RandomForestClassifier()

    param_dist = {"n_estimators"      : randint(5,25),
                  "max_depth"         : randint(1,30),
                  "min_samples_split" : randint(50, 1000),
                  "min_samples_leaf"  : randint(50, 1000),
                  "bootstrap"         : [True, False]}

    n_iter_search = 10
    
    test_preds = dict()
    for train_set,test_set in cv:
        X_temp = X.ix[train_set,:]
        X_test = X.ix[test_set,:]
        y_temp = y.ix[train_set]
        #print(X_temp)
        #,test_set)
        random_search = RandomizedSearchCV(svc_handle, param_distributions=param_dist,
                                           n_iter=n_iter_search,verbose=10,scoring="roc_auc", #["accuracy",]
                                           n_jobs=7,refit=True,cv=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

        random_search_res_xgb = random_search.fit(X_temp, y_temp)
        
        test_preds.update(dict(zip(test_set,random_search_res_xgb.predict_proba(X_test)[:,1])))
    random_search = RandomizedSearchCV(svc_handle, param_distributions=param_dist,
                                           n_iter=n_iter_search,verbose=10,scoring="roc_auc", #["accuracy",]
                                           n_jobs=7,refit=True,cv=cv) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

    random_search_res_xgb = random_search.fit(X, y)
    random_search.feats = X.columns
    rf_model = random_search.best_estimator_
    pickle.dump(random_search,open(outfile,"wb"))
    return(test_preds)
    
def train_nb(X,y,mod_number=1,cv=None):
    svc_handle = GaussianNB()
    
    param_dist = {'priors': [[0.1,0.9],[0.05,0.95],[0.5,0.5],[0.95,0.05],[0.005,0.995]]}
    #svc_handle.fit(X, y)
    n_iter_search = 5
    random_search = RandomizedSearchCV(svc_handle, param_distributions=param_dist,
                                       n_iter=n_iter_search,verbose=10,scoring="roc_auc",
                                       n_jobs=1,cv=cv)

    random_search = random_search.fit(X, y)

    nb_model = random_search.best_estimator_
    #ada_model = random_search_res_ada.best_estimator_
    
    random_search.feats = X.columns
    
    pickle.dump(random_search,open("nb_mod%s.pickle" % (mod_number),"wb"))

def train_nb_cv(X,y,mod_number=1,cv=None,outfile="mod.pickle"):
    svc_handle = GaussianNB()
    
    param_dist = {'priors': [[0.1,0.9],[0.05,0.95],[0.5,0.5],[0.95,0.05],[0.005,0.995]]}
    #svc_handle.fit(X, y)
    n_iter_search = 5

    test_preds = dict()
    for train_set,test_set in cv:
        X_temp = X.ix[train_set,:]
        X_test = X.ix[test_set,:]
        y_temp = y.ix[train_set]
        #print(X_temp)
        #,test_set)
        n_iter_search = 5
        random_search = RandomizedSearchCV(svc_handle, param_distributions=param_dist,
                                           n_iter=n_iter_search,verbose=10,scoring="roc_auc", #["accuracy",]
                                           n_jobs=7,refit=True,cv=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

        random_search_res_xgb = random_search.fit(X_temp, y_temp)
        
        test_preds.update(dict(zip(test_set,random_search_res_xgb.predict_proba(X_test)[:,1])))
        
    random_search = RandomizedSearchCV(svc_handle, param_distributions=param_dist,
                                           n_iter=n_iter_search,verbose=10,scoring="roc_auc", #["accuracy",]
                                           n_jobs=7,refit=True,cv=cv)

    random_search_res_xgb = random_search.fit(X, y)
    random_search.feats = X.columns
    
    pickle.dump(random_search,open(outfile,"wb"))
    
    return(test_preds)

def train_lsvc(X,y,mod_number=1,cv=None):
    min_max_scaler = preprocessing.MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    #X = min_max_scaler.fit_transform(X)
    print(X)
    
    svc_handle = SVC()

    param_dist = {'C': scipy.stats.expon(scale=100), #uniform(0.0,1.0), #,1000,3000], #
                  'gamma': scipy.stats.expon(scale=.1),
                  'kernel': ["rbf"],
                  'class_weight':['balanced', None],                  #,"poly","sigmoid","rbf",
                  'max_iter' : [1000],
                  'probability' : [True]
    }

    n_iter_search = 300
    random_search = RandomizedSearchCV(svc_handle, param_distributions=param_dist,
                                       n_iter=n_iter_search,verbose=10,scoring="roc_auc",
                                       n_jobs=6,cv=cv)

    random_search = random_search.fit(X, y)

    lsvc_model = random_search.best_estimator_
    
    print(random_search.best_score_)
    random_search.feats = X.columns
    random_search.min_max_scaler = min_max_scaler
    #Get the best model that was retrained on all data
    xgb_model = random_search.best_estimator_
    pickle.dump(random_search,open("lsvc_mod%s.pickle" % (mod_number),"wb"))

def train_logreg_cv(X,y,mod_number=1,cv=None,outfile="mod.pickle"):
    """
    Train an XGBoost model with hyper parameter optimization.

    Parameters
    ----------
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
        
    Returns
    -------
    object
        Trained XGBoost model
    object
        Cross-validation results
    """
    
    #xgb_handle = xgb.XGBClassifier()

    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)
    
    #Define distributions to sample from for hyper parameter optimization
    param_dist = {  
        "C": [0.01,0.1,1.0,10.0,50.0,100.0,250.0,500.0,1000.0,2500.0]#from_zero_positive
    }
    
    n_iter_search = 10
    test_preds = dict()
    
    alt_cv = [test_set for train_set,test_set in cv]
    all_feats = list(X.columns)
    selected_feats_overall = []
    best_perf_overall = 0.0

    while len(all_feats) > 10:
        score_dict = {}
        best_perf = 0.0
        for f in all_feats:
            sel_feat = [temp_f for temp_f in all_feats if temp_f != f]
            for ind,test_set in enumerate(alt_cv):
                train_fold_ind = [temp_ind for temp_ind in list(range(len(alt_cv))) if temp_ind != ind]
                train_set = [alt_cv[temp_ind] for temp_ind in train_fold_ind]
                train_set_flat = np.array(list(itertools.chain(*train_set)))
                tot_folds = list(range(len(train_set)))
                cv_temp = []
                
                X_temp = X.ix[train_set_flat,sel_feat]
                y_temp = y.ix[train_set_flat]
                X_test = X.ix[test_set,sel_feat]
                
                old_to_new_index = dict(list(zip(list(X_temp.index),list(range(len(X_temp.index))))))
                
                for tf in tot_folds:
                    fold_flat = copy.deepcopy(list(itertools.chain(*[train_set[tft] for tft in tot_folds if tft != tf])))
                    fold_test = copy.deepcopy(list(train_set[tf]))
                    
                    fold_flat = np.array([old_to_new_index[ff] for ff in fold_flat])
                    fold_test = np.array([old_to_new_index[ff] for ff in fold_test])
                    
                    cv_temp.append((fold_flat,fold_test))
                logreg = LogisticRegression()
                random_search = GridSearchCV(logreg, param_grid=param_dist,verbose=0,scoring="roc_auc", #["accuracy",]
                                                   n_jobs=7,refit=True,cv=cv_temp) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

                random_search_res_xgb = random_search.fit(X_temp,y_temp)
                try:
                    score_dict[f].append(random_search_res_xgb.best_score_)
                except:
                    score_dict[f] = [random_search_res_xgb.best_score_]
                #print(random_search_res_xgb.best_score_,f)
                
                test_preds.update(dict(zip(test_set,random_search_res_xgb.predict_proba(X_test)[:,1])))
        print("=======")
        
        rem_feat = ""

        print(len(all_feats))

        for f_name,scores in score_dict.items():
            avg_perf = sum(score_dict[f_name])/len(score_dict[f_name])
            print(f_name,avg_perf,best_perf)
            if avg_perf > best_perf:
                best_perf = avg_perf
                rem_feat = f_name
        all_feats.remove(rem_feat)
        if best_perf > best_perf_overall:
            selected_feats_overall = all_feats
        print("=======")
    
    test_preds = dict()
    for ind,test_set in enumerate(alt_cv):
        train_fold_ind = [temp_ind for temp_ind in list(range(len(alt_cv))) if temp_ind != ind]
        train_set = [alt_cv[temp_ind] for temp_ind in train_fold_ind]
        train_set_flat = np.array(list(itertools.chain(*train_set)))
        tot_folds = list(range(len(train_set)))
        cv_temp = []
        
        X_temp = X.ix[train_set_flat,selected_feats_overall]
        y_temp = y.ix[train_set_flat]
        X_test = X.ix[test_set,selected_feats_overall]
        
        old_to_new_index = dict(list(zip(list(X_temp.index),list(range(len(X_temp.index))))))
        
        for tf in tot_folds:
            fold_flat = copy.deepcopy(list(itertools.chain(*[train_set[tft] for tft in tot_folds if tft != tf])))
            fold_test = copy.deepcopy(list(train_set[tf]))
            
            fold_flat = np.array([old_to_new_index[ff] for ff in fold_flat])
            fold_test = np.array([old_to_new_index[ff] for ff in fold_test])
            
            cv_temp.append((fold_flat,fold_test))
        logreg = LogisticRegression()
        random_search = GridSearchCV(logreg, param_grid =param_dist,verbose=0,scoring="roc_auc", #["accuracy",]
                                           n_jobs=7,refit=True,cv=cv_temp) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

        random_search_res_xgb = random_search.fit(X_temp,y_temp)
        
        test_preds.update(dict(zip(test_set,random_search_res_xgb.predict_proba(X_test)[:,1])))
    logreg = LogisticRegression()
    random_search = GridSearchCV(logreg, param_grid =param_dist,verbose=10,scoring="roc_auc", #["accuracy",]
                                           n_jobs=7,refit=True,cv=cv)
    
    random_search_res_xgb = random_search.fit(X.ix[:,selected_feats_overall], y[:,selected_feats_overall])
    random_search.feats = X.columns
    pickle.dump(random_search,open(outfile,"wb"))
    return(test_preds)
	
def train_xgb_cv(X,y,mod_number=1,cv=None,outfile="mod.pickle"):
    """
    Train an XGBoost model with hyper parameter optimization.

    Parameters
    ----------
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
        
    Returns
    -------
    object
        Trained XGBoost model
    object
        Cross-validation results
    """
    
    xgb_handle = xgb.XGBClassifier()

    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)
    
    #Define distributions to sample from for hyper parameter optimization
    param_dist = {  
        "n_estimators": [75],#st.randint(5,25), #st.randint(5, 25) #50
        "max_depth": [5], #st.randint(2, 4), #st.randint(3, 8) #6
        #"learning_rate": st.uniform(0.05, 0.4),
        #"colsample_bytree": one_to_left,
        #"subsample": one_to_left,
        #"gamma": st.uniform(0, 10),
        #"reg_alpha": from_zero_positive,
        #"min_child_weight": from_zero_positive#,
        #"scale_pos_weight" : st.uniform(1, 10)
    }
    
    n_iter_search = 1
    test_preds = dict()
    
    alt_cv = [test_set for train_set,test_set in cv]
    all_feats = list(X.columns)
    selected_feats_overall = []
    best_perf_overall = 0.0

    while len(all_feats) > 2:
        score_dict = {}
        best_perf = 0.0
        for f in all_feats:
            sel_feat = [temp_f for temp_f in all_feats if temp_f != f]
            for ind,test_set in enumerate(alt_cv):
                train_fold_ind = [temp_ind for temp_ind in list(range(len(alt_cv))) if temp_ind != ind]
                train_set = [alt_cv[temp_ind] for temp_ind in train_fold_ind]
                train_set_flat = np.array(list(itertools.chain(*train_set)))
                tot_folds = list(range(len(train_set)))
                cv_temp = []
                
                X_temp = X.ix[train_set_flat,sel_feat]
                y_temp = y.ix[train_set_flat]
                X_test = X.ix[test_set,sel_feat]
                
                old_to_new_index = dict(list(zip(list(X_temp.index),list(range(len(X_temp.index))))))
                
                for tf in tot_folds:
                    fold_flat = copy.deepcopy(list(itertools.chain(*[train_set[tft] for tft in tot_folds if tft != tf])))
                    fold_test = copy.deepcopy(list(train_set[tf]))
                    
                    fold_flat = np.array([old_to_new_index[ff] for ff in fold_flat])
                    fold_test = np.array([old_to_new_index[ff] for ff in fold_test])
                    
                    cv_temp.append((fold_flat,fold_test))
                
                random_search = RandomizedSearchCV(xgb_handle, param_distributions=param_dist,
                                                   n_iter=n_iter_search,verbose=0,scoring="roc_auc", #["accuracy",]
                                                   n_jobs=1,refit=True,cv=cv_temp) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

                random_search_res_xgb = random_search.fit(X_temp,y_temp)
                try:
                    score_dict[f].append(random_search_res_xgb.best_score_)
                except:
                    score_dict[f] = [random_search_res_xgb.best_score_]
                #print(random_search_res_xgb.best_score_,f)
                
                test_preds.update(dict(zip(test_set,random_search_res_xgb.predict_proba(X_test)[:,1])))
        print("=======")
        
        rem_feat = ""

        print(len(all_feats))

        for f_name,scores in score_dict.items():
            avg_perf = sum(score_dict[f_name])/len(score_dict[f_name])
            print(f_name,avg_perf,best_perf)
            if avg_perf > best_perf:
                best_perf = avg_perf
                rem_feat = f_name
        all_feats.remove(rem_feat)
        if best_perf > best_perf_overall:
            best_perf_overall = best_perf
            selected_feats_overall = copy.deepcopy(all_feats)
        print("=======")

    param_dist = {  
        "n_estimators": [75],#st.randint(5,25), #st.randint(5, 25) #50
        "max_depth": st.randint(3, 8), #6
        "learning_rate": st.uniform(0.05, 0.4),
        #"colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        "reg_alpha": from_zero_positive,
        #"min_child_weight": from_zero_positive,
        "scale_pos_weight" : st.uniform(1, 5)
    }
    
    n_iter_search = 50
	
    test_preds = dict()
    for ind,test_set in enumerate(alt_cv):
        train_fold_ind = [temp_ind for temp_ind in list(range(len(alt_cv))) if temp_ind != ind]
        train_set = [alt_cv[temp_ind] for temp_ind in train_fold_ind]
        train_set_flat = np.array(list(itertools.chain(*train_set)))
        tot_folds = list(range(len(train_set)))
        cv_temp = []
        
        X_temp = X.ix[train_set_flat,selected_feats_overall]
        y_temp = y.ix[train_set_flat]
        X_test = X.ix[test_set,selected_feats_overall]
        
        old_to_new_index = dict(list(zip(list(X_temp.index),list(range(len(X_temp.index))))))
        
        for tf in tot_folds:
            fold_flat = copy.deepcopy(list(itertools.chain(*[train_set[tft] for tft in tot_folds if tft != tf])))
            fold_test = copy.deepcopy(list(train_set[tf]))
            
            fold_flat = np.array([old_to_new_index[ff] for ff in fold_flat])
            fold_test = np.array([old_to_new_index[ff] for ff in fold_test])
            
            cv_temp.append((fold_flat,fold_test))
        
        random_search = RandomizedSearchCV(xgb_handle, param_distributions=param_dist,
                                           n_iter=n_iter_search,verbose=0,scoring="roc_auc", #["accuracy",]
                                           n_jobs=1,refit=True,cv=cv_temp) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

        random_search_res_xgb = random_search.fit(X_temp,y_temp)
        
        test_preds.update(dict(zip(test_set,random_search_res_xgb.predict_proba(X_test)[:,1])))
    

    random_search = RandomizedSearchCV(xgb_handle, param_distributions=param_dist,
                                           n_iter=n_iter_search,verbose=0,scoring="roc_auc", #["accuracy",]
                                           n_jobs=1,refit=True,cv=cv)
    
    random_search_res_xgb = random_search.fit(X.ix[:,selected_feats_overall], y)
    random_search.feats = X.columns
    pickle.dump(random_search,open(outfile,"wb"))
    return(test_preds)
    
def train_xgb(X,y,mod_number=1,cv=None,outfile="ensemble.pickle"):
    """
    Train an XGBoost model with hyper parameter optimization.

    Parameters
    ----------
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
        
    Returns
    -------
    object
        Trained XGBoost model
    object
        Cross-validation results
    """
    
    xgb_handle = xgb.XGBClassifier()

    one_to_left = st.beta(10, 1)  
    from_zero_positive = st.expon(0, 50)
    
    #Define distributions to sample from for hyper parameter optimization
    param_dist = {  
        "n_estimators": st.randint(5, 150),
        "max_depth": st.randint(5, 10),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        "reg_alpha": from_zero_positive,
        "min_child_weight": from_zero_positive,
    }

    n_iter_search = 250
    random_search = RandomizedSearchCV(xgb_handle, param_distributions=param_dist,
                                       n_iter=n_iter_search,verbose=0,scoring="roc_auc", #["accuracy",]
                                       n_jobs=1,refit=True,cv=cv) # StratifiedKFold(n_splits=5, shuffle=True,random_state=42))

    random_search_res_xgb = random_search.fit(X, y)
    
    print(random_search_res_xgb.best_params_)
    preds = []
    y_vals = []
    for train_indexes,test_indexes in cv:
        train_indexes = list(train_indexes)
        test_indexes = list(test_indexes)
        
        x_train_temp = X.ix[train_indexes,:]
        y_train_temp = y.ix[train_indexes]
        x_test_temp = X.ix[test_indexes,:]
        y_test_temp = y.ix[test_indexes]
            
        xgb_handle_cross = xgb.XGBClassifier(**random_search_res_xgb.best_params_)
        xgb_handle_cross.fit(x_train_temp,y_train_temp)
        preds.extend(xgb_handle_cross.predict_proba(x_test_temp)[:,1])
        y_vals.extend(y_test_temp)
        
    fpr, tpr, _ = roc_curve(y_vals,preds)

    roc_auc = auc(fpr,tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Secrify kaggle competition')
    #plt.legend(loc="lower right")
    plt.show()

    print("Best blend score:")
    print(random_search_res_xgb.best_score_)
    random_search.feats = X.columns
    #Get the best model that was retrained on all data
    xgb_model = random_search.best_estimator_
    pickle.dump(random_search,open(outfile,"wb"))
    #return(xgb_model,random_search_res_xgb)

def div_into_folds(identifiers,n_folds=2):
    folds = [[] for y in range(n_folds)]
    for i in range(0,len(identifiers),n_folds):
        r = random.randint(0,n_folds-1)
        folds[r].append(identifiers[i])

        folds_assign = list(range(n_folds))[r+1:]
        folds_assign.extend(list(range(n_folds))[0:r])
        i_next_to = i + 1
        for fa in folds_assign:
            try: folds[fa].append(identifiers[i_next_to])
            except IndexError: continue
            i_next_to += 1
    return(folds)
    
def divide_folds(main_path="",
                 output_train="data/Sc_train_identifiers.txt",
                 output_test="data/Sc_test_identifiers.txt",
                 input_enriched="data/Sc_resultstable_enriched.txt",
                 output_train_instances_pos="data/Sc_resultstable_enriched_train.txt",
                 output_test_instances_pos="data/Sc_resultstable_enriched_test.txt",
                 input_depleted="data/Sc_resultstable_depleted.txt",  
                 output_train_instances_neg="data/Sc_resultstable_depleted_train.txt",
                 output_test_instances_neg="data/Sc_resultstable_depleted_test.txt",
                 random_seed=42):

    random.seed(random_seed) 
    locs_enriched,gene_ident_to_row_ident_enriched = get_set_all_aa(os.path.join(main_path,input_enriched))
    locs_depleted,gene_ident_to_row_ident_depleted = get_set_all_aa(os.path.join(main_path,input_depleted))

    gene_ident_length = [[k,len(v)] for k,v in gene_ident_to_row_ident_enriched.items()]
    gene_ident_length = sorted(gene_ident_length, key=operator.itemgetter(1), reverse=True)
    gene_idents_sorted = [g for g,l in gene_ident_length]

    uniq_depleted = list(set(gene_ident_to_row_ident_depleted.keys()) - set(gene_idents_sorted))

    gene_idents_sorted.extend(uniq_depleted)

    folds = div_into_folds(gene_idents_sorted,n_folds=10)

    new_folds = []
    new_folds.append(list(itertools.chain(*folds[:7])))
    new_folds.append(list(itertools.chain(*folds[7:])))
    folds = new_folds

    outfile_train = open(os.path.join(main_path,output_train),"w")
    outfile_train.write("\n".join(folds[0]))
    outfile_train.close()

    outfile_test = open(os.path.join(main_path,output_test),"w")
    outfile_test.write("\n".join(folds[1]))
    outfile_test.close()

    folds = map(set,folds)
    
    infile = open(os.path.join(main_path,input_enriched))
    outfile_enriched_train = open(os.path.join(main_path,output_train_instances_pos),"w")
    outfile_enriched_test = open(os.path.join(main_path,output_test_instances_pos),"w")
    for line in infile:
        if line.startswith("identifier"):
            outfile_enriched_train.write(line.strip()+",class\n")
            outfile_enriched_test.write(line.strip()+",class\n")
            continue

        split_line = line.strip().split(",")
        if split_line[1] in folds[0]:
            outfile_enriched_train.write("Sc"+line.strip()+",1\n")
        elif split_line[1] in folds[1]:
            #print("YES")
            outfile_enriched_test.write("Sc"+line.strip()+",1\n")

    infile = open(os.path.join(main_path,input_depleted))
    outfile_depleted_train = open(os.path.join(main_path,output_train_instances_neg),"w")
    outfile_depleted_test = open(os.path.join(main_path,output_test_instances_neg),"w")
    for line in infile:
        if line.startswith("identifier"):
            outfile_depleted_train.write(line.strip()+",class\n")
            outfile_depleted_test.write(line.strip()+",class\n")
            continue

        split_line = line.strip().split(",")
        if split_line[1] in folds[0]:
            outfile_depleted_train.write("Sc"+line.strip()+",0\n")
        elif split_line[1] in folds[1]:
            outfile_depleted_test.write("Sc"+line.strip()+",0\n")

def stat_analysis(main_path="",
                  input_enriched="data/train_enriched.csv",
                  input_depleted="data/train_depleted.csv"):
                  
    locs_enriched,gene_ident_to_row_ident_enriched = get_set_all_aa(os.path.join(main_path,input_enriched),ret_duplicated_pos=True)
    locs_depleted,gene_ident_to_row_ident_depleted = get_set_all_aa(os.path.join(main_path,input_depleted),ret_duplicated_pos=True)
    counted_locs_depleted = Counter(locs_depleted)
    num_overlap = []
    for le in locs_enriched:
        try: num_overlap.append(counted_locs_depleted[le])
        except KeyError: num_overlap.append(0)
    
    plt.hist(num_overlap,bins=300)
    plt.title("Distribution of non-unique overlap between amino acids")
    ax = plt.gca()
    ax.set_xlabel('Amino acid overlap (#) with depleted')
    ax.set_ylabel('Frequency (#)')
    plt.savefig(os.path.join(main_path,"figs/hist_aa_overlap.pdf"), format='pdf')
    plt.close()
                     
    locs_enriched,gene_ident_to_row_ident_enriched = get_set_all_aa(os.path.join(main_path,input_enriched))
    locs_depleted,gene_ident_to_row_ident_depleted = get_set_all_aa(os.path.join(main_path,input_depleted))
    distrib_overlap,agg_seqs,num_total,num_in_other = get_set_all_aa_compare(os.path.join(main_path,input_enriched),locs_depleted)

    non_overlapping_seqs = []
    for seq in agg_seqs:
        if len(seq) == 0: continue
        prev_pos = False
        temp_seq = ""
        for aa,pos in seq:
            if prev_pos:
                if pos-prev_pos < 2:
                    temp_seq += aa
                else:
                    non_overlapping_seqs.append(temp_seq)
                    temp_seq = aa
                    prev_pos = pos
            else:
                if len(temp_seq) > 0:
                    non_overlapping_seqs.append(temp_seq)
                temp_seq = aa
                prev_pos = pos
    print("".join(non_overlapping_seqs))
    print(Counter("".join(non_overlapping_seqs)))
    print("Fragments in enriched: %s" % (len(list(itertools.chain(*list(gene_ident_to_row_ident_enriched.values()))))))
    print("Fragments in depleted: %s" % (len(list(itertools.chain(*list(gene_ident_to_row_ident_depleted.values()))))))
    print("Unique genes in enriched: %s" % (len(gene_ident_to_row_ident_enriched.keys())))
    print("Unique genes in depleted: %s" % (len(gene_ident_to_row_ident_depleted.keys())))
    print("Total number of positions enriched: %s" % (len(locs_enriched)))
    print("Total number of positions depleted: %s" % (len(locs_depleted)))
    print("Total number of positions overlap: %s" % (len(locs_depleted.intersection(locs_enriched))))
    print("Number of fragments: %s" % (num_total))
    print("Number of fragments that have overlap: %s" % (num_in_other))
    print("Percentage of positions overlap: %s" % ((len(locs_depleted.intersection(locs_enriched))/float(len(locs_enriched)))*100))


    plt.hist(distrib_overlap,bins=20)
    plt.title("Distribution of overlap between enriched fragments with depleted fragments")
    plt.savefig(os.path.join(main_path,"figs/hist_dist_overlap.pdf"), format='pdf')
    plt.close()

    gene_idents = list(gene_ident_to_row_ident_enriched.keys())
    gene_idents.extend(list(gene_ident_to_row_ident_depleted.keys()))
    gene_idents = list(set(gene_idents))

def get_feats(main_path="",infile_name="data/train.csv",assign_class=0):    
    libs_prop = get_libs_aa(os.path.join(main_path,"expasy/"))
    #print(libs_prop)
    feats = list(libs_prop.keys())
    
    data_df_depleted,rownames_depleted = get_feats_simple(os.path.join(main_path,infile_name),libs_prop,assign_class=assign_class)
    
    infile = open("feat_pssm.csv")
    pssm_feats = {}
    for line in infile:
        if line.startswith("identifier"):
            header = ["pssm|"+h for h in line.strip().split(",")]
            continue
        splitline = line.strip().split(",")
        pssm_feats[splitline[0]] = dict(zip(header[1:],map(float,splitline[1:])))
    
    all_pssm = []
    for idx,idt in enumerate(rownames_depleted):
        #print(idx)
        try:
            all_pssm.append(pssm_feats[idt])
            working = pssm_feats[idt]
        except:
            all_pssm.append(dict(zip(list(working.keys()),[0]*len(list(working.keys())))))
        
        
    all_seqs = []
    all_seqs.extend(get_all_seqs(os.path.join(main_path,infile_name)))

    order_aa = get_all_poss_aa(all_seqs)
    
    #seq_feats = []
    seq_feats_fragments = []
    for seq in all_seqs:
        #seq_feats.append(dict(zip(order_aa,count_aa(seq,aa_order=order_aa))))
        #seq_ten_splits = list(split_seq(seq,3))
        #seq_ten_splits.extend([seq[0:5],seq[-5:]])
        seq_ten_splits = [seq[0:5],seq[-5:],seq[0:10],seq[-10:],seq[-20:],seq[0:20],seq[0],seq[1],seq[2],seq[3],seq[4],seq[5],seq[-1],seq[-2],seq[-3],seq[-4],seq[-5],seq[-6]]
        instance_dict = {}
        instance_dict.update(dict(zip(order_aa,count_aa(seq,aa_order=order_aa))))
        instance_dict["seq_length"] = len(seq)
        #seq_ten_splits.extend([seq[0:5],seq[-5:]])
        
        for index,spl_s in enumerate(seq_ten_splits):
            temp_dict = get_feats_simple_seq(spl_s,libs_prop)
            temp_dict = dict([[k+"|"+str(index),v] for k,v in temp_dict.items()])
            instance_dict.update(temp_dict)
        
        seq_feats_fragments.append(instance_dict)

    feats.extend(list(seq_feats_fragments[0].keys()))
    feats.extend(list(working.keys()))


    data_df = []
    #data_df.extend(data_df_enriched)
    data_df.extend(data_df_depleted)
    
    new_df = []
    
    for row_df,row_seq in zip(data_df,seq_feats_fragments):
        row_df.update(row_seq)
            
    for row_df,row_seq in zip(data_df,all_pssm):
        if sum(row_seq.values()) == 0:
            #del data_df[row_df[0]]
            continue
        row_df.update(row_seq)
        new_df.append(row_df)
    
    return(new_df,feats)

def get_preds(main_path="C:/Users/asus/Dropbox/secretome/kaggle/"):
    test_df,feats = get_feats()
    #print(test_df)
    test_df = pd.DataFrame(test_df)
    #print(test_df)
    
    test_pd = pd.read_csv("data/test.csv")
    test_df.index = test_pd["identifier"]
    
    #print(test_df)
    preds_df = {}
    for filename in os.listdir(main_path):
        if filename.endswith(".pickle"):
            #if filename != "mod0.pickle": continue
            selected_mod = pickle.load(open(os.path.join(main_path,filename),"rb"))
            if "nb_" in filename: continue
            try: feats = selected_mod.feats
            except: continue
            preds_df[filename.rstrip(".pickle")] = selected_mod.predict_proba(test_df[feats])[:,1]
            test_df[filename.rstrip(".pickle")] = selected_mod.predict_proba(test_df[feats])[:,1]
            #feats.append(
    
    preds_df = pd.DataFrame(preds_df)
    test_pd = pd.read_csv("data/test.csv")
    preds_df.index = test_pd["identifier"]
    #test_df["prediction"] = preds_df["mod0"]
    test_df["med_prediction"] = preds_df.median(axis=1)
    test_df["mean_prediction"] = preds_df.mean(axis=1)
    #preds_df["mod0"]
    test_df.to_csv("submission.csv")
    plt.hist(preds_df.std(axis=1),bins=500)
    plt.show()
    #print(preds_df.std(axis=1))
    plt.hist(preds_df.median(axis=1),bins=500)
    plt.show()
    plt.hist(preds_df.mean(axis=1),bins=500)
    plt.show()
#    print(preds_df)

def get_preds_ensemble(main_path="C:/Users/asus/Dropbox/secretome/kaggle/"):
    test_df,feats = get_feats(infile_name="data/test.csv")
    test_pd = pd.read_csv("data/test.csv")
    #print(test_df)
    test_df = pd.DataFrame(test_df,index=list(test_pd["identifier"]))
    
    #print(test_df)
    preds_df = {}

    all_feats = list(test_df.columns)
    all_feats = [f for f in all_feats if f not in ["identifier","class","Ensembl_geneID"]]
    for filename in os.listdir(main_path):
        if filename.endswith(".pickle"):
            if filename == "ensemble.pickle": continue
            print(filename)
            selected_mod = pickle.load(open(os.path.join(main_path,filename),"rb"))
            try: feats = selected_mod.feats
            except: continue
            preds_df[filename.rstrip(".pickle")] = selected_mod.predict_proba(test_df[feats])[:,1]
            test_df[filename.rstrip(".pickle")] = selected_mod.predict_proba(test_df[feats])[:,1]
            all_feats.append(filename.rstrip(".pickle"))

    #feats_ensemble = [f for f in all_feats if "|" not in f]
    selected_mod = pickle.load(open(os.path.join(main_path,"ensemble.pickle"),"rb"))
    feats = selected_mod.feats
    ensemble_pred = selected_mod.predict_proba(test_df[feats])[:,1]

    preds_df = pd.DataFrame(preds_df)
    test_pd = pd.read_csv("data/test.csv")
    preds_df.index = test_pd["identifier"]
    test_df["prediction"] = ensemble_pred
    test_df["med_prediction"] = preds_df.median(axis=1)
    test_df["mean_prediction"] = preds_df.mean(axis=1)
    #preds_df["mod0"]
    test_df.to_csv("submission.csv")
    plt.hist(preds_df.std(axis=1),bins=500)
    plt.show()
    #print(preds_df.std(axis=1))
    plt.hist(preds_df.median(axis=1),bins=500)
    plt.show()
    plt.hist(preds_df.mean(axis=1),bins=500)
    plt.show()
    
def get_preds(main_path="C:/Users/asus/Dropbox/secretome/kaggle/"):
    test_df,feats = get_feats(infile_name="data/test.csv")
    #print(test_df)
    test_df = pd.DataFrame(test_df)
    #print(test_df)
    
    test_pd = pd.read_csv("data/test.csv")
    test_df.index = test_pd["identifier"]
    
    #print(test_df)
    preds_df = {}
    for filename in os.listdir(main_path):
        if filename.endswith(".pickle"):
            #if filename != "mod0.pickle": continue
            selected_mod = pickle.load(open(os.path.join(main_path,filename),"rb"))
            if "nb_" in filename: continue
            try: feats = selected_mod.feats
            except: continue
            preds_df[filename.rstrip(".pickle")] = selected_mod.predict_proba(test_df[feats])[:,1]
    preds_df = pd.DataFrame(preds_df)
    test_pd = pd.read_csv("data/test.csv")
    preds_df.index = test_pd["identifier"]
    test_df["prediction"] = preds_df["mod0"]
    test_df["med_prediction"] = preds_df.median(axis=1)
    test_df["mean_prediction"] = preds_df.mean(axis=1)
    #preds_df["mod0"]
    test_df.to_csv("submission.csv")
    plt.hist(preds_df.std(axis=1),bins=500)
    plt.show()
    #print(preds_df.std(axis=1))
    plt.hist(preds_df.median(axis=1),bins=500)
    plt.show()
    plt.hist(preds_df.mean(axis=1),bins=500)
    plt.show()
#    print(preds_df)

def make_CV(folds,ident_to_indices):
    cv = []
    for index_fold,fold in enumerate(folds):
        train = itertools.chain(*[fold_temp for index_temp,fold_temp in enumerate(folds) if index_fold != index_temp])
        test = folds[index_fold]
        
        train_indices = []
        for ens_id in train:
            try: train_indices.extend(ident_to_indices[ens_id])
            except: continue
        
        test_indices = []
        for ens_id in test:
            try: test_indices.extend(ident_to_indices[ens_id])
            except: continue
            
        cv.append((np.array(sorted(train_indices)),np.array(sorted(test_indices))))
    return(cv)

def train_all_instances(main_path="",input_depleted="data/train_depleted.csv",input_enriched="data/train_enriched.csv",random_seed=42):
    data_df_enriched,feats = get_feats(infile_name=input_enriched,assign_class=1)
    data_df_depleted,feats = get_feats(infile_name=input_depleted,assign_class=0)

    data_df = []
    data_df.extend(data_df_enriched)
    data_df.extend(data_df_depleted)
    
    random.seed(random_seed)         
    locs_enriched,gene_ident_to_row_ident_enriched = get_set_all_aa(os.path.join(main_path,input_enriched))
    locs_depleted,gene_ident_to_row_ident_depleted = get_set_all_aa(os.path.join(main_path,input_depleted))
    
    gene_ident_length = [[k,len(v)] for k,v in gene_ident_to_row_ident_enriched.items()]
    gene_ident_length = sorted(gene_ident_length, key=operator.itemgetter(1), reverse=True)
    gene_idents_sorted = [g for g,l in gene_ident_length]
    
    uniq_depleted = list(set(gene_ident_to_row_ident_depleted.keys()) - set(gene_idents_sorted))
    
    gene_idents_sorted.extend(uniq_depleted)

    folds = div_into_folds(gene_idents_sorted,n_folds=5)
    
    pd_df = pd.DataFrame(data_df)
    
    ident_to_indices = {}
    for index,ens_id in enumerate(list(pd_df["Ensembl_geneID"])):
        try:
            ident_to_indices[ens_id].append(index)
        except:
            ident_to_indices[ens_id] = [index]
    
    cv = make_CV(folds,ident_to_indices)

    feat_groups = list(set([f.split("|")[0] for f in feats if len(f) > 3 and f != "seq_length"]))
    feat_groups = [fg for fg in feat_groups if fg != "pssm"]
    feat_aa = list(set([f.split("|")[0] for f in feats if len(f) < 3]))
    pred_feats = {}
    for index,f_group in enumerate(feat_groups):
        #index = 1
        #print(f_group)
        selected_feats = [f for f in feats if f.startswith(f_group)]
        selected_feats.append("seq_length")
        #selected_feats.append("organism")
        selected_feats.extend(feat_aa)
        
        X = pd_df[selected_feats]
        y = pd_df["class"]
        
        #train_xgb(X,y,mod_number=index,cv=cv,outfile="mod_xgb_%s.pickle" % (index))
        #preds = train_logreg_cv(X,y,mod_number=index,cv=cv,outfile="mod_xgb_%s.pickle" % (index))
        preds = train_xgb_cv(X,y,mod_number=index,cv=cv,outfile="mod_xgb_%s.pickle" % (index))
        print("DONE!")
        preds = sorted(list(preds.items()),key=operator.itemgetter(0))
        preds = [j for i,j in preds]
        for instance,pred in zip(data_df,preds):
            instance["mod_xgb_%s" % (index)] = pred
        feats.append("mod_xgb_%s" % (index))
        
        """
        preds = train_ada_cv(X,y,mod_number=index,cv=cv,outfile="mod_ada_%s.pickle" % (index))
        preds = sorted(list(preds.items()),key=operator.itemgetter(0))
        preds = [j for i,j in preds]
        for instance,pred in zip(data_df,preds):
            instance["mod_ada_%s" % (index)] = pred
        feats.append("mod_ada_%s" % (index))
        
        preds = train_rf_cv(X,y,mod_number=index,cv=cv,outfile="mod_rf_%s.pickle" % (index))
        preds = sorted(list(preds.items()),key=operator.itemgetter(0))
        preds = [j for i,j in preds]
        for instance,pred in zip(data_df,preds):
            instance["mod_rf_%s" % (index)] = pred
        feats.append("mod_rf_%s" % (index))
        
        preds = train_nb_cv(X,y,mod_number=index,cv=cv,outfile="mod_nb_%s.pickle" % (index))
        preds = sorted(list(preds.items()),key=operator.itemgetter(0))
        preds = [j for i,j in preds]
        pred_feats["mod_nb_%s" % (index)] = preds
        for instance,pred in zip(data_df,preds):
            instance["mod_nb_%s" % (index)] = pred
        feats.append("mod_nb_%s" % (index))
        """
        
    #print(index,f_group)
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y,preds)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    input()
    """
    pd_df = pd.DataFrame(data_df)
    print(list(pd_df))
    feats_ensemble = [f for f in feats if "|" not in f]
    X = pd_df[feats_ensemble]
    y = pd_df["class"]
    
    train_xgb(X,y,mod_number=1,cv=cv,outfile="ensemble.pickle")
    
    print(feat_groups)
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_0"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_1"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_2"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_3"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_4"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_5"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_6"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_7"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_8"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_9"])
    print(auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y, X["mod_xgb_10"])
    print(auc(fpr, tpr))
    

if __name__ == "__main__":
    #get_all_seqs("data/test.csv")
    main_path=""
    #stat_analysis(main_path=main_path)
    #divide_folds(main_path=main_path)
    train_all_instances(main_path=main_path)
    #get_preds()
    #get_preds_ensemble()