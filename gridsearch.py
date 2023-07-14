import os
import csv
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve



alert_data = pd.read_csv(
    "dataset/alert.csv",
    index_col=0,
    encoding="utf-8"
)

netflow_fields = pd.read_csv("dataset/netflow/netflow-feats.csv").columns

flows_list = os.listdir("dataset/netflow/flow")
np.random.shuffle(flows_list)

sample_size = 1000
sample_list = flows_list[:sample_size]
test_list = flows_list[sample_size:]


ident_features = [
    'src_ip',
    'dst_ip',
    'src_port',
    'dst_port'
]
port_features = [
    'src_port',
    'dst_port'
]
ip_features = [
    'ip_proto_pkt_cnt',
    'ip_proto_bytes_cnt',
]
udp_features = [
   # 'udp_proto_pkt_cnt',
    'udp_proto_bytes_cnt',
]
tcp_features = [
   # 'tcp_proto_pkt_cnt',
    'tcp_proto_bytes_cnt',
    'tcp_proto_syn_cnt',
    'tcp_proto_fin_cnt',
    'tcp_proto_ack_cnt',
    'tcp_proto_psh_cnt',
    'tcp_proto_rst_cnt',
    'tcp_proto_synack_cnt',
]
icmp_features = [
   # 'icmp_proto_pkt_cnt',
    'icmp_proto_bytes_cnt',
    'icmp_proto_echo_request_cnt',
    'icmp_proto_echo_reply_cnt'
]
dns_features = [
    'dns_query_cnt',
    'dns_response_cnt'
]
feats = (
      port_features
    + ip_features
    + tcp_features
    + udp_features
    + icmp_features
    + dns_features
)

timestamp_feats = [
    'first_alert_time_sec',
    'last_alert_time_sec',
    'first_pkt_time_sec',
    'last_pkt_time_sec'
]


balanced_batches = []
batch_size = 50
print("start...", end='')

param_grid = {
    'max_depth' : [None, 5,10, 20, 30, 40],
    'n_estimators' : [50, 100, 150, 200, 250,300],
}

xgb = XGBClassifier()

grid_search = GridSearchCV(
    xgb,
    param_grid,
    scoring=['f1','precision','recall'],
    refit='f1',
    cv=2,
    n_jobs=6,
    verbose=10
)

for L in [1,10,20,30,40,50,60,70,80,90]:
    for batch_count in range(len(sample_list) // batch_size):
        print(f"\rbatch n. {batch_count} initialization...", end='')
        batch = pd.concat(
            [
                pd.concat(
                    [
                        pd.read_csv(
                            f"dataset/netflow/flow/{filename}",
                            names=netflow_fields,
                            encoding="cp1251"
                        ) for filename in sample_list[s:s+5]
                    ],
                    ignore_index=True
                ).merge(
                    alert_data,
                    how='left',
                    on=ident_features,
                    indicator=True
                ) [feats + timestamp_feats + ['_merge']]
                for s in range(batch_size * (batch_count    ),
                            batch_size * (batch_count + 1), 5)
            ],
            ignore_index=True
        )
        batch['is_alert'] = batch['_merge'].replace(
            {
                'both': 1, 'left_only': 0, 'right_only': 0
            },
        )
        batch = batch[
            (
                (
                    batch.is_alert == 1
                ) & (
                    batch.first_alert_time_sec >= batch.first_pkt_time_sec
                ) & (
                    batch.first_alert_time_sec <= batch.last_pkt_time_sec
                ) & (
                    batch.last_alert_time_sec <= batch.last_pkt_time_sec
                )
            ) | (
                batch.is_alert == 0
            )
        ][feats + ['is_alert']]

        alert_index = batch[batch.is_alert == 1].index

        balanced_batch_index = np.concatenate(
            [
                np.random.choice(
                    batch[batch.is_alert == 0].index,
                    size=alert_index.shape[0] * L,
                    replace=False
                ),
                alert_index
            ],
            axis=0
        )
        balanced_batches.append(batch.loc[balanced_batch_index])

    X = pd.concat(
        balanced_batches,
        sort=False,
        ignore_index=True
    ).drop_duplicates()

    y = X['is_alert']
    X = X[feats]


    grid_search.fit(X, y)

    print(f"\nDisb: 1:{L} \
            Fact disb: 1:{X[y==0].shape[0]/X[y==1].shape[0]}\
            Best param: {grid_search.best_params_}\
            Best score: {grid_search.best_score_}")
    csv.DictWriter(
            open(f"./grid_search_1:{L}.csv", 'w', newline=''),
        fieldnames=grid_search.best_params_.keys()
    ).writerows([grid_search.best_params_])
