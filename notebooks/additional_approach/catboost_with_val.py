import pandas as pd
from catboost import Pool, CatBoostRanker
from copy import deepcopy

##
posi_train = pd.read_csv('/home/igor/PycharmProjects/bert-qa/notebooks/additional_approach/positive.csv')
neg_train = pd.read_csv('/home/igor/PycharmProjects/bert-qa/notebooks/additional_approach/negative.csv')

posi_val = pd.read_csv('/home/igor/PycharmProjects/bert-qa/notebooks/additional_approach/positive_val.csv')
neg_val = pd.read_csv('/home/igor/PycharmProjects/bert-qa/notebooks/additional_approach/negative_val.csv')
##
train_df = pd.concat([posi_train, neg_train], axis=0)
val_df = pd.concat([posi_val, neg_val], axis=0)
##
train_df.drop_duplicates(inplace=True)
val_df.drop_duplicates(inplace=True)
train_df.sort_values(by='id', inplace=True)
val_df.sort_values(by='id', inplace=True)
##

y_train = train_df['target']
group_train = train_df['id']
X_train = train_df.drop(['target', 'id'], axis=1)

y_val = val_df['target']
group_val = val_df['id']
X_val = val_df.drop(['target', 'id'], axis=1)
##
train = Pool(
    data=X_train,
    label=y_train,
    group_id=group_train
)

test = Pool(
    data=X_val,
    label=y_val,
    group_id=group_val
)
##
default_parameters = {
    'iterations': 2000,
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
    'verbose': True,
    'random_seed': 0,
}

parameters = {}
##

def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=test):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function
    parameters['thread_count'] = -1

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool)

    return model

##


model = fit_model('RMSE', {'custom_metric': ['PrecisionAt:top=10', 'RecallAt:top=10', 'MAP:top=10']},)
