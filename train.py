import os

import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

if not os.path.exists('./model'):
    os.makedirs('./model')

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


train_balanced_batches = []
test_balanced_batches = []

batch_size = 50

train_disbalances = [1,10,20,30,40,50]
test_disbalances = [1,10,20,30,40,50]


for batch_count in range(len(sample_list) // batch_size):
    print(f"train batch n. {batch_count} initialization...", end='\r')
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


    alert_index = batch[batch.is_alert == 1].index

    balanced_batch_index = np.concatenate(
        [
            np.random.choice(
                batch[batch.is_alert == 0].index,
                size=alert_index.shape[0] * 100,
                replace=False
            ),
            alert_index
        ],
        axis=0
    )
    train_balanced_batches.append(batch.loc[balanced_batch_index])


for batch_count in range(len(test_list) // batch_size):
    print(f"test batch n. {batch_count} initialization...", end='\r')
    batch = pd.concat(
        [
            pd.concat(
                [
                    pd.read_csv(
                        f"dataset/netflow/flow/{filename}",
                        names=netflow_fields,
                        encoding="cp1251"
                    ) for filename in test_list[s:s+5]
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


    alert_index = batch[batch.is_alert == 1].index

    balanced_batch_index = np.concatenate(
        [
            np.random.choice(
                batch[batch.is_alert == 0].index,
                size=alert_index.shape[0] * 100,
                replace=False
            ),
            alert_index
        ],
        axis=0
    )
    test_balanced_batches.append(batch.loc[balanced_batch_index])

for train_disbalance in train_disbalances:   
    plt.clf()    
    X = pd.concat(
        train_balanced_batches,
        sort=False,
        ignore_index=True
    ).drop_duplicates()

    X = X.loc[ 
        np.concatenate(
            [
                np.random.choice(
                    X[X.is_alert==0].index,
                    size=X[X.is_alert==1].shape[0] * train_disbalance,
                    replace=False
                ),
                X[X.is_alert==1].index
            ],
            axis=0
        )
    ]
    
    y = X.is_alert
    X = X[feats]

    print(f"train disbalance: {X[y==1].shape[0] / X[y==0].shape[0]}")

    ne, md = 50, 10

    xgb_big = XGBClassifier(
        n_estimators=ne,
        max_depth=md,
        learning_rate=0.1,
        subsample=0.8,
        n_jobs=-1,
        verbosity=2
    )

    xgb_big.fit(X, y)
    xgb_big.save_model(f"./model/xgb_nestim-{ne}_mdepth-{md}_disbalance-{train_disbalance}.cbm")
    pd.DataFrame(feats).to_csv(f"./model/feats_xgb_nestim-{ne}_mdepth-{md}_disbalance-{train_disbalance}.csv")

    for test_disbalance in test_disbalances:
       
        X_test = pd.concat(
            test_balanced_batches,
            sort=False,
            ignore_index=True
        )

        X_test = X_test.loc[ 
            np.concatenate(
                [
                    np.random.choice(
                        X_test[X_test.is_alert == 0].index,
                        size=X_test[X_test.is_alert==1].shape[0] * test_disbalance,
                        replace=False
                    ),
                    X_test[X_test.is_alert==1].index
                ],
                axis=0
            )
        ]
        
        y_test = X_test.is_alert
        X_test = X_test[feats]

        y_true = y_test
        y_scores = xgb_big.predict_proba(X_test)[:,1]

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)


        plt.grid(True, color='grey', linestyle='--', linewidth=1)
        plt.title(f'prc train-1:{train_disbalance}')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(recall, precision, linewidth=2, label=f"test-1:{test_disbalance}")
        plt.legend()
        print(f"test aucpr for {train_disbalance}:{test_disbalance} = {auc(recall, precision)}")
    plt.savefig(f"pic/prc_xgb_nestim-{ne}_mdepth-{md}_train-1:{train_disbalance}.png")
    
