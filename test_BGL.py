"""
This file contains functions for performing running fair regression
algorithms and the set of baseline methods.

See end of file to see sample use of running fair regression.
"""
import numpy as np
import pandas as pd
import data_parser as parser
from sklearn.model_selection import train_test_split
from fairlearn.reductions import BoundedGroupLoss, SquareLoss, ExponentiatedGradient
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from fairlearn.metrics import MetricFrame

TEST_SIZE = 0.4  # fraction of observations from each protected group
Theta = np.linspace(0, 1.0, 41)
alpha = (Theta[1] - Theta[0])/2
DATA_SPLIT_SEED = 4
_SMALL = True


def train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED):
    """Split the input dataset into train and test sets

    TODO: Need to make sure both train and test sets have enough
    observations from each subgroup
    """
    # size of the training data
    groups = list(a.unique())
    x_train_sets = {}
    x_test_sets = {}
    y_train_sets = {}
    y_test_sets = {}
    a_train_sets = {}
    a_test_sets = {}

    for g in groups:
        x_g = x[a == g]
        a_g = a[a == g]
        y_g = y[a == g]
        x_train_sets[g], x_test_sets[g], a_train_sets[g], a_test_sets[g], y_train_sets[g], y_test_sets[g] = train_test_split(x_g, a_g, y_g, test_size=TEST_SIZE, random_state=random_seed)

    x_train = pd.concat(x_train_sets.values())
    x_test = pd.concat(x_test_sets.values())
    y_train = pd.concat(y_train_sets.values())
    y_test = pd.concat(y_test_sets.values())
    a_train = pd.concat(a_train_sets.values())
    a_test = pd.concat(a_test_sets.values())

    # resetting the index
    x_train.index = range(len(x_train))
    y_train.index = range(len(y_train))
    a_train.index = range(len(a_train))
    x_test.index = range(len(x_test))
    y_test.index = range(len(y_test))
    a_test.index = range(len(a_test))
    return x_train, a_train, y_train, x_test, a_test, y_test


# Global Variables
Metric = r2_score
x, a, y = parser.clean_communities_full()
#x, a, y = parser.clean_lawschool_full()
#dat = pd.read_csv('data/genetc_sim.csv')
#x = dat.loc[:, ['x1', 'x2', 'x3']]
#y = dat.loc[:, ['y']]
#a = dat.loc[:, ['race']].squeeze()
#x['race'] = a
x_train, a_train, y_train, x_test, a_test, y_test = train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)
lm = LinearRegression().fit(x_train, y_train)
y_train_pred_lm = lm.predict(x_train)
y_test_pred_lm = lm.predict(x_test)
mse_frame_lm_train = MetricFrame(metrics=Metric, y_true=y_train, y_pred=y_train_pred_lm, sensitive_features=a_train)
mse_frame_lm_test = MetricFrame(metrics=Metric, y_true=y_test, y_pred=y_test_pred_lm, sensitive_features=a_test)
resuld_df = pd.DataFrame({"method":['lm'], "train_group1":[mse_frame_lm_train.by_group[0]], "train_group2":[mse_frame_lm_train.by_group[1]], "train_overall":[mse_frame_lm_train.overall],
                          "test_group1":[mse_frame_lm_test.by_group[0]], "test_group2":[mse_frame_lm_test.by_group[1]], "test_overall":[mse_frame_lm_test.overall]})

lm = LinearRegression().fit(x_train.loc[a_train==0], y_train.loc[a_train==0])
y_train_pred_lm = lm.predict(x_train)
y_test_pred_lm = lm.predict(x_test)
mse_frame_lm_train = MetricFrame(metrics=Metric, y_true=y_train, y_pred=y_train_pred_lm, sensitive_features=a_train)
mse_frame_lm_test = MetricFrame(metrics=Metric, y_true=y_test, y_pred=y_test_pred_lm, sensitive_features=a_test)

df2 = pd.DataFrame({"method":['lm_dat1'], "train_group1":[mse_frame_lm_train.by_group[0]], "train_group2":[mse_frame_lm_train.by_group[1]], "train_overall":[mse_frame_lm_train.overall],
                          "test_group1":[mse_frame_lm_test.by_group[0]], "test_group2":[mse_frame_lm_test.by_group[1]], "test_overall":[mse_frame_lm_test.overall]})
resuld_df = resuld_df.append(df2, ignore_index=True)

lm = LinearRegression().fit(x_train.loc[a_train==1], y_train.loc[a_train==1])
y_train_pred_lm = lm.predict(x_train)
y_test_pred_lm = lm.predict(x_test)
mse_frame_lm_train = MetricFrame(metrics=Metric, y_true=y_train, y_pred=y_train_pred_lm, sensitive_features=a_train)
mse_frame_lm_test = MetricFrame(metrics=Metric, y_true=y_test, y_pred=y_test_pred_lm, sensitive_features=a_test)
df2 = pd.DataFrame({"method":['lm_dat2'], "train_group1":[mse_frame_lm_train.by_group[0]], "train_group2":[mse_frame_lm_train.by_group[1]], "train_overall":[mse_frame_lm_train.overall],
                          "test_group1":[mse_frame_lm_test.by_group[0]], "test_group2":[mse_frame_lm_test.by_group[1]], "test_overall":[mse_frame_lm_test.overall]})
resuld_df = resuld_df.append(df2, ignore_index=True)

Upper_bound = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
for i in range(Upper_bound.shape[0]):
    bgl = BoundedGroupLoss(SquareLoss(0, 1), upper_bound=Upper_bound[i])
    mitigator = ExponentiatedGradient(LinearRegression(), bgl)
    mitigator.fit(X=x_train, y=y_train, sensitive_features=a_train)
    y_train_pred = mitigator.predict(x_train)
    y_test_pred = mitigator.predict(x_test)
    mse_frame_train = MetricFrame(metrics=Metric, y_true=y_train, y_pred=y_train_pred,
                                  sensitive_features=a_train)
    mse_frame_test = MetricFrame(metrics=Metric, y_true=y_test, y_pred=y_test_pred,
                                 sensitive_features=a_test)
    df2 = pd.DataFrame({"method":[Upper_bound[i]], "train_group1":[mse_frame_train.by_group[0]], "train_group2":[mse_frame_train.by_group[1]], "train_overall":[mse_frame_train.overall],
                          "test_group1":[mse_frame_test.by_group[0]], "test_group2":[mse_frame_test.by_group[1]], "test_overall":[mse_frame_test.overall]})
    resuld_df = resuld_df.append(df2, ignore_index=True)







############# Simulated data

# Global Variables
Metric = mean_squared_error
dat = pd.read_csv('data/genetc_sim.csv')
x = dat.loc[:, ['x1', 'x2', 'x3']]
y = dat.loc[:, ['y']]
a = dat.loc[:, ['race']].squeeze()
#x['race'] = a
x_train, a_train, y_train, x_test, a_test, y_test = train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)
lm = LinearRegression().fit(x_train, y_train)
y_train_pred_lm = lm.predict(x_train)
y_test_pred_lm = lm.predict(x_test)
mse_frame_lm_train = MetricFrame(metrics=Metric, y_true=y_train, y_pred=y_train_pred_lm, sensitive_features=a_train)
mse_frame_lm_test = MetricFrame(metrics=Metric, y_true=y_test, y_pred=y_test_pred_lm, sensitive_features=a_test)
resuld_df = pd.DataFrame({"method":['lm'], "train_group1":[mse_frame_lm_train.by_group[0]], "train_group2":[mse_frame_lm_train.by_group[1]], "train_overall":[mse_frame_lm_train.overall],
                          "test_group1":[mse_frame_lm_test.by_group[0]], "test_group2":[mse_frame_lm_test.by_group[1]], "test_overall":[mse_frame_lm_test.overall]})

lm = LinearRegression().fit(x_train.loc[a_train==0], y_train.loc[a_train==0])
y_train_pred_lm = lm.predict(x_train)
y_test_pred_lm = lm.predict(x_test)
mse_frame_lm_train = MetricFrame(metrics=Metric, y_true=y_train, y_pred=y_train_pred_lm, sensitive_features=a_train)
mse_frame_lm_test = MetricFrame(metrics=Metric, y_true=y_test, y_pred=y_test_pred_lm, sensitive_features=a_test)

df2 = pd.DataFrame({"method":['lm_dat1'], "train_group1":[mse_frame_lm_train.by_group[0]], "train_group2":[mse_frame_lm_train.by_group[1]], "train_overall":[mse_frame_lm_train.overall],
                          "test_group1":[mse_frame_lm_test.by_group[0]], "test_group2":[mse_frame_lm_test.by_group[1]], "test_overall":[mse_frame_lm_test.overall]})
resuld_df = resuld_df.append(df2, ignore_index=True)

lm = LinearRegression().fit(x_train.loc[a_train==1], y_train.loc[a_train==1])
y_train_pred_lm = lm.predict(x_train)
y_test_pred_lm = lm.predict(x_test)
mse_frame_lm_train = MetricFrame(metrics=Metric, y_true=y_train, y_pred=y_train_pred_lm, sensitive_features=a_train)
mse_frame_lm_test = MetricFrame(metrics=Metric, y_true=y_test, y_pred=y_test_pred_lm, sensitive_features=a_test)
df2 = pd.DataFrame({"method":['lm_dat2'], "train_group1":[mse_frame_lm_train.by_group[0]], "train_group2":[mse_frame_lm_train.by_group[1]], "train_overall":[mse_frame_lm_train.overall],
                          "test_group1":[mse_frame_lm_test.by_group[0]], "test_group2":[mse_frame_lm_test.by_group[1]], "test_overall":[mse_frame_lm_test.overall]})
resuld_df = resuld_df.append(df2, ignore_index=True)

Upper_bound = np.array([0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25])
for i in range(Upper_bound.shape[0]):
    bgl = BoundedGroupLoss(SquareLoss(0, 1), upper_bound=Upper_bound[i])
    mitigator = ExponentiatedGradient(LinearRegression(), bgl)
    mitigator.fit(X=x_train, y=y_train, sensitive_features=a_train)
    y_train_pred = mitigator.predict(x_train)
    y_test_pred = mitigator.predict(x_test)
    mse_frame_train = MetricFrame(metrics=Metric, y_true=y_train, y_pred=y_train_pred,
                                  sensitive_features=a_train)
    mse_frame_test = MetricFrame(metrics=Metric, y_true=y_test, y_pred=y_test_pred,
                                 sensitive_features=a_test)
    df2 = pd.DataFrame({"method":[Upper_bound[i]], "train_group1":[mse_frame_train.by_group[0]], "train_group2":[mse_frame_train.by_group[1]], "train_overall":[mse_frame_train.overall],
                          "test_group1":[mse_frame_test.by_group[0]], "test_group2":[mse_frame_test.by_group[1]], "test_overall":[mse_frame_test.overall]})
    resuld_df = resuld_df.append(df2, ignore_index=True)

a = 1