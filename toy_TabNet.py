import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import missingno as msno
from sklearn.cluster import DBSCAN

from sklearn.datasets import make_blobs
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split

# config
np.random.seed(2123)
mv_percent = 10  # percentage of missing values created in the dataset

# create cluster dataset
features, clusters = make_blobs(n_samples=[50, 100, 50],
                                centers=[[1, 5, 1], [5, 8, 5], [8, 2, 3]],
                                n_features=3,
                                cluster_std=1.5, )

# plot dataset
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=clusters)
plt.show()

# create missing values
features = pd.DataFrame(features, columns=[f"F{x + 1}" for x in range(features.shape[1])])
clusters = pd.DataFrame(clusters, columns=["C"])

features_mv = features.copy()
features_mv["F3"] = features_mv["F3"].mask(np.random.random(features.shape[0]) < mv_percent / 100)

msno.matrix(features_mv)
plt.show()

# use TabNet to predict missing values
test = features_mv[features_mv["F3"].isna()].drop("F3", axis=1).to_numpy()
train = features_mv[features_mv["F3"].notna()]
train_features = train.iloc[:, :-1].to_numpy()
train_target = train.iloc[:, -1].to_numpy().reshape(-1, 1)

x_train, x_val, y_train, y_val = train_test_split(train_features, train_target, test_size=0.75)

model = TabNetRegressor(optimizer_fn=torch.optim.Adam,
                        optimizer_params=dict(lr=2e-2),
                        scheduler_params={"step_size": 50, "gamma": 0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type='entmax')

model.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],
    eval_metric=['rmse'],
    max_epochs=200, patience=20,
)

# fill original dataset with predictions
features_filled = features_mv.copy()
features_filled[features_filled["F3"].isna()] = model.predict(test)

# apply clustering
dbscan = DBSCAN(eps=2, min_samples=10).fit(features_filled)
clusters_filled = dbscan.labels_

# plot new clustering
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(features_filled.iloc[:, 0], features_filled.iloc[:, 1], features_filled.iloc[:, 2], c=clusters_filled)
plt.show()
