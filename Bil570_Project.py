# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 00:36:04 2019

@author: Ali
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt # Plot Learning curve
from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn import svm
import statsmodels.api as sm # p-values
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve # Learning Curve
from lightgbm import plot_metric
warnings.filterwarnings("ignore")

def prepare(df):
    global json_cols
    global train_dict

    df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[ (df['release_year'] <= 19) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[ (df['release_year'] > 19)  & (df['release_year'] < 100), "release_year"] += 1900
    
    releaseDate = pd.to_datetime(df['release_date']) 
    df['release_dayofweek'] = releaseDate.dt.dayofweek 
    df['release_quarter'] = releaseDate.dt.quarter     
    
    rating_na = df.groupby(["release_year","original_language"])['rating'].mean().reset_index()
    df[df.rating.isna()]['rating'] = df.merge(rating_na, how = 'left' ,on = ["release_year","original_language"])
    vote_count_na = df.groupby(["release_year","original_language"])['totalVotes'].mean().reset_index()
    df[df.totalVotes.isna()]['totalVotes'] = df.merge(vote_count_na, how = 'left' ,on = ["release_year","original_language"])
    df['rating'] = df['rating'].fillna(1.5)
    df['totalVotes'] = df['totalVotes'].fillna(6)
    df['weightedRating'] = ( df['rating']*df['totalVotes'] + 6.367 * 1000 ) / ( df['totalVotes'] + 1000 )


    df['originalBudget'] = df['budget']
    df['inflationBudget'] = df['budget'] + df['budget']*1.8/100*(2018-df['release_year']) #Inflation simple formula
    df['budget'] = np.log1p(df['budget']) 
    
    
    # Thanks to this Kernel for the next 7 features https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation
    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
    df['_collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
    le = LabelEncoder()
    le.fit(list(df['_collection_name'].fillna('')))
    df['_collection_name'] = le.transform(df['_collection_name'].fillna('').astype(str))
    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

    
    
    df['_popularity_mean_year'] = df['popularity'] / df.groupby("release_year")["popularity"].transform('mean')
    df['_budget_runtime_ratio'] = df['budget']/df['runtime'] 
    df['_budget_popularity_ratio'] = df['budget']/df['popularity']
    df['_budget_year_ratio'] = df['budget']/(df['release_year']*df['release_year'])
    df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']

    df['_popularity_totalVotes_ratio'] = df['totalVotes']/df['popularity']
    df['_rating_popularity_ratio'] = df['rating']/df['popularity']
    df['_rating_totalVotes_ratio'] = df['totalVotes']/df['rating']
    df['_totalVotes_releaseYear_ratio'] = df['totalVotes']/df['release_year']
    df['_budget_rating_ratio'] = df['budget']/df['rating']
    df['_runtime_rating_ratio'] = df['runtime']/df['rating']
    df['_budget_totalVotes_ratio'] = df['budget']/df['totalVotes']
    
    df['has_homepage'] = 1
    df.loc[pd.isnull(df['homepage']) ,"has_homepage"] = 0
    
    df['isbelongs_to_collectionNA'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']) ,"isbelongs_to_collectionNA"] = 1
    
    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0 ,"isTaglineNA"] = 1 

    df['isOriginalLanguageEng'] = 0 
    df.loc[ df['original_language'] == "en" ,"isOriginalLanguageEng"] = 1
    
    df['isTitleDifferent'] = 1
    df.loc[ df['original_title'] == df['title'] ,"isTitleDifferent"] = 0 

    df['isMovieReleased'] = 1
    df.loc[ df['status'] != "Released" ,"isMovieReleased"] = 0 

    # get collection id
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x : np.nan if len(x)==0 else x[0]['id'])
    
    df['original_title_letter_count'] = df['original_title'].str.len() 
    df['original_title_word_count'] = df['original_title'].str.split().str.len() 


    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()
    
    df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))
    df['cast_count'] = df['cast'].apply(lambda x : len(x))
    df['crew_count'] = df['crew'].apply(lambda x : len(x))
    

    df['meanruntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')
    df['meanPopularityByYear'] = df.groupby("release_year")["popularity"].aggregate('mean')
    df['meanBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('mean')
    df['meantotalVotesByYear'] = df.groupby("release_year")["totalVotes"].aggregate('mean')
    df['meanTotalVotesByRating'] = df.groupby("rating")["totalVotes"].aggregate('mean')
    df['medianBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('median')

    for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies'] :
        df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col+'_etc' for n in [d['name'] for d in x]])))).map(lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)
    df.drop(['genres_etc'], axis = 1, inplace = True)
    
    df = df.drop(['id', 'revenue','belongs_to_collection','genres','homepage','imdb_id','overview','runtime'
    ,'poster_path','production_companies','production_countries','release_date','spoken_languages'
    ,'status','title','Keywords','cast','crew','original_language','original_title','tagline', 'collection_id'
    ],axis=1)
    
    df.fillna(value=0.0, inplace = True) 

    return df
train = pd.read_csv('train.csv')

train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning
train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          
train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs
train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven
train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 
train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty
train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood
train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada
train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times
train.loc[train['id'] == 1007,'budget'] = 2              # Zyzzyx Road 
train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   
train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 
train.loc[train['id'] == 1542,'budget'] = 1              # All at Once
train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II
train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit
train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon
train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
train.loc[train['id'] == 1885,'budget'] = 12             # In the Cut
train.loc[train['id'] == 2091,'budget'] = 10             # Deadfall
train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
train.loc[train['id'] == 2491,'budget'] = 6              # Never Talk to Strangers
train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams
train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture
train.loc[train['id'] == 335,'budget'] = 2 
train.loc[train['id'] == 348,'budget'] = 12
train.loc[train['id'] == 470,'budget'] = 13000000 
train.loc[train['id'] == 513,'budget'] = 1100000
train.loc[train['id'] == 640,'budget'] = 6 
train.loc[train['id'] == 696,'budget'] = 1
train.loc[train['id'] == 797,'budget'] = 8000000 
train.loc[train['id'] == 850,'budget'] = 1500000
train.loc[train['id'] == 1199,'budget'] = 5 
train.loc[train['id'] == 1282,'budget'] = 9               # Death at a Funeral
train.loc[train['id'] == 1347,'budget'] = 1
train.loc[train['id'] == 1755,'budget'] = 2
train.loc[train['id'] == 1801,'budget'] = 5
train.loc[train['id'] == 1918,'budget'] = 592 
train.loc[train['id'] == 2033,'budget'] = 4
train.loc[train['id'] == 2118,'budget'] = 344 
train.loc[train['id'] == 2252,'budget'] = 130
train.loc[train['id'] == 2256,'budget'] = 1 
train.loc[train['id'] == 2696,'budget'] = 10000000





test = pd.read_csv('test.csv')

#Clean Data
test.loc[test['id'] == 6733,'budget'] = 5000000
test.loc[test['id'] == 3889,'budget'] = 15000000
test.loc[test['id'] == 6683,'budget'] = 50000000
test.loc[test['id'] == 5704,'budget'] = 4300000
test.loc[test['id'] == 6109,'budget'] = 281756
test.loc[test['id'] == 7242,'budget'] = 10000000
test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family
test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage
test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee
test.loc[test['id'] == 3033,'budget'] = 250 
test.loc[test['id'] == 3051,'budget'] = 50
test.loc[test['id'] == 3084,'budget'] = 337
test.loc[test['id'] == 3224,'budget'] = 4  
test.loc[test['id'] == 3594,'budget'] = 25  
test.loc[test['id'] == 3619,'budget'] = 500  
test.loc[test['id'] == 3831,'budget'] = 3  
test.loc[test['id'] == 3935,'budget'] = 500  
test.loc[test['id'] == 4049,'budget'] = 995946 
test.loc[test['id'] == 4424,'budget'] = 3  
test.loc[test['id'] == 4460,'budget'] = 8  
test.loc[test['id'] == 4555,'budget'] = 1200000 
test.loc[test['id'] == 4624,'budget'] = 30 
test.loc[test['id'] == 4645,'budget'] = 500 
test.loc[test['id'] == 4709,'budget'] = 450 
test.loc[test['id'] == 4839,'budget'] = 7
test.loc[test['id'] == 3125,'budget'] = 25 
test.loc[test['id'] == 3142,'budget'] = 1
test.loc[test['id'] == 3201,'budget'] = 450
test.loc[test['id'] == 3222,'budget'] = 6
test.loc[test['id'] == 3545,'budget'] = 38
test.loc[test['id'] == 3670,'budget'] = 18
test.loc[test['id'] == 3792,'budget'] = 19
test.loc[test['id'] == 3881,'budget'] = 7
test.loc[test['id'] == 3969,'budget'] = 400
test.loc[test['id'] == 4196,'budget'] = 6
test.loc[test['id'] == 4221,'budget'] = 11
test.loc[test['id'] == 4222,'budget'] = 500
test.loc[test['id'] == 4285,'budget'] = 11
test.loc[test['id'] == 4319,'budget'] = 1
test.loc[test['id'] == 4639,'budget'] = 10
test.loc[test['id'] == 4719,'budget'] = 45
test.loc[test['id'] == 4822,'budget'] = 22
test.loc[test['id'] == 4829,'budget'] = 20
test.loc[test['id'] == 4969,'budget'] = 20
test.loc[test['id'] == 5021,'budget'] = 40 
test.loc[test['id'] == 5035,'budget'] = 1 
test.loc[test['id'] == 5063,'budget'] = 14 
test.loc[test['id'] == 5119,'budget'] = 2 
test.loc[test['id'] == 5214,'budget'] = 30 
test.loc[test['id'] == 5221,'budget'] = 50 
test.loc[test['id'] == 4903,'budget'] = 15
test.loc[test['id'] == 4983,'budget'] = 3
test.loc[test['id'] == 5102,'budget'] = 28
test.loc[test['id'] == 5217,'budget'] = 75
test.loc[test['id'] == 5224,'budget'] = 3 
test.loc[test['id'] == 5469,'budget'] = 20 
test.loc[test['id'] == 5840,'budget'] = 1 
test.loc[test['id'] == 5960,'budget'] = 30
test.loc[test['id'] == 6506,'budget'] = 11 
test.loc[test['id'] == 6553,'budget'] = 280
test.loc[test['id'] == 6561,'budget'] = 7
test.loc[test['id'] == 6582,'budget'] = 218
test.loc[test['id'] == 6638,'budget'] = 5
test.loc[test['id'] == 6749,'budget'] = 8 
test.loc[test['id'] == 6759,'budget'] = 50 
test.loc[test['id'] == 6856,'budget'] = 10
test.loc[test['id'] == 6858,'budget'] =  100
test.loc[test['id'] == 6876,'budget'] =  250
test.loc[test['id'] == 6972,'budget'] = 1
test.loc[test['id'] == 7079,'budget'] = 8000000
test.loc[test['id'] == 7150,'budget'] = 118
test.loc[test['id'] == 6506,'budget'] = 118
test.loc[test['id'] == 7225,'budget'] = 6
test.loc[test['id'] == 7231,'budget'] = 85
test.loc[test['id'] == 5222,'budget'] = 5
test.loc[test['id'] == 5322,'budget'] = 90
test.loc[test['id'] == 5350,'budget'] = 70
test.loc[test['id'] == 5378,'budget'] = 10
test.loc[test['id'] == 5545,'budget'] = 80
test.loc[test['id'] == 5810,'budget'] = 8
test.loc[test['id'] == 5926,'budget'] = 300
test.loc[test['id'] == 5927,'budget'] = 4
test.loc[test['id'] == 5986,'budget'] = 1
test.loc[test['id'] == 6053,'budget'] = 20
test.loc[test['id'] == 6104,'budget'] = 1
test.loc[test['id'] == 6130,'budget'] = 30
test.loc[test['id'] == 6301,'budget'] = 150
test.loc[test['id'] == 6276,'budget'] = 100
test.loc[test['id'] == 6473,'budget'] = 100
test.loc[test['id'] == 6842,'budget'] = 30


test['revenue'] = np.nan

# features from https://www.kaggle.com/kamalchhirang/eda-simple-feature-engineering-external-data
train = pd.merge(train, pd.read_csv('TrainAdditionalFeatures.csv'), how='left', on=['imdb_id'])
test = pd.merge(test, pd.read_csv('TestAdditionalFeatures.csv'), how='left', on=['imdb_id'])


#train = pd.merge(train, additionalTrainData, how='left', on=['imdb_id'],axis=1)
train['revenue'] = np.log1p(train['revenue'])
y = train['revenue'].values

json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

for col in tqdm(json_cols + ['belongs_to_collection']) :
    train[col] = train[col].apply(lambda x : get_dictionary(x))
    test[col] = test[col].apply(lambda x : get_dictionary(x))
    
def get_json_dict(df) :
    global json_cols
    result = dict()
    for e_col in json_cols :
        d = dict()
        rows = df[e_col].values
        for row in rows :
            if row is None : continue
            for i in row :
                if i['name'] not in d :
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result

train_dict = get_json_dict(train)
test_dict = get_json_dict(test)

# remove cateogry with bias and low frequency
for col in json_cols :
    
    remove = []
    train_id = set(list(train_dict[col].keys()))
    test_id = set(list(test_dict[col].keys()))   
    
    remove += list(train_id - test_id) + list(test_id - train_id)
    for i in train_id.union(test_id) - set(remove) :
        if train_dict[col][i] < 5 or i == '' :
            remove += [i]
            
    for i in remove :
        if i in train_dict[col] :
            del train_dict[col][i]
        if i in test_dict[col] :
            del test_dict[col][i]
            
all_data = prepare(pd.concat([train, test]).reset_index(drop = True))
train = all_data.loc[:train.shape[0] - 1,:]
test = all_data.loc[train.shape[0]:,:] 

X = train
X = X.replace([np.inf,-np.inf],0)
test = test.replace([np.inf,-np.inf],0)

X['mean_samp'] = X.mean(axis=1)
X['var_samp'] = np.log1p(X.var(axis=1))
X['sum_samp'] = X.sum(axis=1)

test['mean_samp'] = test.mean(axis=1)
test['var_samp'] = np.log1p(test.var(axis=1))
test['sum_samp'] = test.sum(axis=1)
#------------------------------------------------------------------------------
# Feature Selection
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_BE = cols

X_new = pd.DataFrame()    
test_new = pd.DataFrame()
for i in range(len(selected_features_BE)):
    X_new[selected_features_BE[i]] = X[selected_features_BE[i]]
    test_new[selected_features_BE[i]] = test[selected_features_BE[i]]    
#------------------------------------------------------------------------------
kfold = 10
skf = KFold(n_splits=kfold, shuffle=True,random_state=None)

test1 = pd.read_csv('test.csv')
test_id = test1.id.values

sub = pd.DataFrame()
subb = pd.DataFrame()
sub['target'] = np.zeros_like(test_id)
subb['valid'] = np.zeros_like(X_new.budget.values)

sub1 = pd.DataFrame()
subb1 = pd.DataFrame()
sub1['target'] = np.zeros_like(test_id)
subb1['valid'] = np.zeros_like(X_new.budget.values)

sub2 = pd.DataFrame()
subb2 = pd.DataFrame()
sub2['target'] = np.zeros_like(test_id)
subb2['valid'] = np.zeros_like(X_new.budget.values)
fold = list(KFold(10, shuffle = True, random_state = None).split(X_new))

hyper_params = {'max_depth' : [3, 5, 8, 10],
                'min_child_weight' : [3, 5, 7, 10],
                }
                
grid = GridSearchCV(xgb.XGBRegressor(),hyper_params,n_jobs=-1,verbose=10,cv=fold)

grid.fit(X_new,y)


params = {'min_child_weight': grid.best_params_['min_child_weight'], 
          'eta': 0.04, 'colsample_bytree': 0.8,
          'max_depth': grid.best_params_['max_depth'], 'subsample': 0.75,
          'lambda': 2, 'nthread': -1, 
          'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,
          'eval_metric': 'rmse', 'objective': 'binary:logistic'}

hyper_params1 = {'max_depth' : [3, 5, 7, 10],
                'learning_rate' : [0.001, 0.005,0.01]}
                
grid1 = GridSearchCV(lgb.LGBMRegressor(),hyper_params1,n_jobs=-1,verbose=10,cv=fold)

grid1.fit(X_new,y)


params1 = {'n_estimators' : 5000, 'objective' : 'regression', 
          'metric' : 'rmse', 'max_depth' : grid1.best_params_['max_depth'],
                             'num_leaves':30, 
                             'min_child_samples':100,
                             'learning_rate':grid1.best_params_['learning_rate'],
                             'boosting' : 'gbdt',
                             'min_data_in_leaf': 10,
                             'feature_fraction' : 0.9,
                             'bagging_freq' : 1,
                             'bagging_fraction' : 0.9,
                             'importance_type':'gain',
                             'lambda_l1' : 0.2,
                             'bagging_seed':2019, 
                             'subsample':.8, 
                             'colsample_bytree':.9,
                             'use_best_model':True}

tolerances = [0.001, 0.01, 0.01, 1, 10, 100]
    
    
Gammas = [0.001, 0.01, 0.1, 1]
parameters_of_gridsearch = {'C' : tolerances, 'gamma' : Gammas}
    
grid_search_optimization = GridSearchCV(estimator = svm.SVR(),
                                        param_grid = parameters_of_gridsearch, 
                                        cv = fold, n_jobs=-1,verbose=10)
grid_search_optimization.fit(X_new, y)

evals_result = {}
evals_result1 = {}

for i, (train_index, test_index) in enumerate(fold):
    X_train = X_new.loc[train_index, :]
    y_train = y[train_index]
    X_valid = X_new.loc[test_index, :]
    y_valid = y[test_index]
    
    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test_new)
    d_train_val = xgb.DMatrix(X_new)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    
    mdl = xgb.train(params, d_train, 5000, watchlist, early_stopping_rounds=500,
                    maximize=False, verbose_eval=100)

    # Predict on our test data
    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
    v_train = mdl.predict(d_train_val, ntree_limit=mdl.best_ntree_limit)
    sub['target'] += p_test/10
    subb['valid'] += v_train/10
    
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)
    test_data = lgb.Dataset(test_new)
    
    model = lgb.train(params1, train_data, 5000, valid_sets=[validation_data],
                      early_stopping_rounds = 500, evals_result = evals_result,
                      verbose_eval=100)
    
    test_pred = model.predict(test_new)
    valid_pred = model.predict(X_new)
    
    sub1['target'] += test_pred/10
    subb1['valid'] += valid_pred/10
    
    svm_model = svm.SVR(C = grid_search_optimization.best_params_['C'],
                        gamma = grid_search_optimization.best_params_['gamma'],
                        kernel = 'rbf',verbose = 100)

    svm_model.fit(X_train, y_train)
    
    test_pred_svm = svm_model.predict(test_new)
    valid_pred_svm = svm_model.predict(X_new)
    
    sub2['target'] += test_pred_svm/10
    subb2['valid'] += valid_pred_svm/10
    

submit = pd.DataFrame({'id': test1.id, 'revenue':np.expm1(sub['target'])})
submit.to_csv('submission_new.csv', index=False)

#fine_tune_train = pd.DataFrame()
#fine_tune_train['xgb'] = subb['valid']
##fine_tune_train['lxgb'] = subb1['valid']
##fine_tune_train['svm'] = subb2['valid']
#
#fine_tune_test = pd.DataFrame()
#fine_tune_test['xgb'] = sub['target']
##fine_tune_test['lxgb'] = sub1['target']
##fine_tune_test['svm'] = sub2['target']
#
#fold1 = list(KFold(10, shuffle = True, random_state = None).split(fine_tune_train))
#
#hyper_params = {'max_depth' : [3, 5, 8, 10, 15],
#                'min_child_weight' : [3, 5, 7, 10],
#                'learning_rate' : [0.07, 0.1, 0.15]
#                }
#                
#grid4 = GridSearchCV(xgb.XGBRegressor(),hyper_params,n_jobs=-1,verbose=10,cv=fold1)
#
#grid4.fit(fine_tune_train,y)
#
#reg = xgb.XGBRegressor(max_depth = grid4.best_params_['max_depth'], 
#                       min_child_weight = grid4.best_params_['min_child_weight'],
#                       learning_rate = grid4.best_params_['learning_rate'], colsample_bylevel = 0.8, 
#                       subsample = 0.75, reg_lambda = 2, nthread = -1,
#                       booster = 'gbtree', silent = 1, gamma = 0)
#
#reg.fit(fine_tune_train,y)
#
#y_pred = reg.predict(fine_tune_test)

plot_metric(evals_result1, metric='rmse')

plot_metric(evals_result, metric='rmse')

xgb.plot_importance(mdl)
lgb.plot_importance(model)

train_size = np.linspace(0.1, 1.0, 20)
plt.figure()
plt.title("Learning Curve for SVM")
plt.xlabel("Traning Set Size")
plt.ylabel("Error")
train_sizes, train_scores, test_scores = learning_curve(
        svm_model, X_train, y_train, cv=10, train_sizes=train_size, 
        scoring = 'neg_mean_squared_error')

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
plt.grid()

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

plt.legend(loc="best")