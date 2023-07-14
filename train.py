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


balanced_batches = []
batch_size = 50
print("start...", end='')

disbalance = [1,10,20,30,40,50,60,70,80,90]

for train_disbalanse in disbalance:
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
                    size=alert_index.shape[0] * train_disbalanse,
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

    print(f"\ndisbalance: {X[y==1].shape[0] / X[y==0].shape[0]}")

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
    xgb_big.save_model(f"./models/xgb_nestim-{ne}_mdepth-{md}_disbalance-{train_disbalanse}.cbm")
    pd.DataFrame(feats).to_csv(f"./models/feats/xgb_nestim-{ne}_mdepth-{md}_disbalance-{train_disbalanse}.csv")

    for test_disbalance in disbalance:
        balanced_batches = []
        batch_size = 50
        for batch_count in range(len(test_list) // batch_size):
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
                    for s in range(batch_size * (batch_count)   ,
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
                        size=alert_index.shape[0] * test_disbalance,
                        replace=False
                    ),
                    alert_index
                ],
                axis=0
            )
            balanced_batches.append(batch.loc[balanced_batch_index])

        X_test = pd.concat(
            balanced_batches,
            sort=False,
            ignore_index=True
        )
        y_test = X_test.is_alert
        X_test = X_test[feats]

        y_true = y_test
        y_scores = xgb_big.predict_proba(X_test)[:,1]

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)


        plt.grid(True, color='grey', linestyle='--', linewidth=1)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall, precision, linewidth=2, label=f"1:{test_disbalance}")
        plt.legend()
        plt.savefig(f"pic/prc_xgb_nestim-{ne}_mdepth-{md}_train-1:{train_disbalanse}_test-1:{test_disbalance}.png")
        print(f"test aucpr for {train_disbalanse}:{test_disbalance} = {auc(recall, precision)}")
