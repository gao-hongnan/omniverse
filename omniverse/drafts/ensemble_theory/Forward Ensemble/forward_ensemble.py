import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import roc_auc_score


class ForwardEnsemble:
    def __init__(
        self,
        dir: str,
        oof: pd.DataFrame,
        weight_interval: int,
        patience: int,
        min_increase: float,
        target_column_names: List[str],
        pred_column_names: List[str],
    ):
        super().__init__()
        self.dir = dir
        FILES = os.listdir(dir)
        self.oof_list = np.sort([f for f in FILES if "oof" in f])
        self.num_oofs = len(self.oof_list)

        self.oof = oof  # the oof csv with n rows m columns where n is the number of images in the dataset, and m be the number of target columns * number of oof you have
        self.weight_interval = weight_interval
        self.patience = patience
        self.min_increase = min_increase
        self.target_column_names = (
            target_column_names  # target_cols = oof[0].iloc[:, 1:12].columns.tolist()
        )
        self.pred_column_names = (
            pred_column_names  # pred_cols = oof[0].iloc[:, 15:].columns.tolist()
        )

        self.col_len = len(target_column_names)

        self.num_test_images = len(oof[0])

        # get ground truth
        self.y_true = oof[0][target_column_names].values

        self.all_oof_preds = np.zeros(
            (self.num_test_images, self.num_oofs * self.col_len)
        )

        # append all oof preds to all_oof_preds: for example - k=0 -> all_oof_preds[:,0:11] = self.oof[0][['ETT - Abnormal OOF', etc]].values
        for k in range(self.num_oofs):
            self.all_oof_preds[
                :,
                int(k * self.col_len) : int((k + 1) * self.col_len),
            ] = oof[k][pred_column_names].values

        self.model_i_score, self.model_i_index, self.model_i_weight = 0, 0, 0

    def __len__(self):
        return len(
            self.column_names
        )  # get number of prediction columns, in multi-label, should have more than 1 column, while in binary, there is only 1

    def macro_multilabel_auc(self, label, pred):
        """ Also works for binary AUC like Melanoma"""
        aucs = []
        for i in range(self.col_len):
            aucs.append(roc_auc_score(label[:, i], pred[:, i]))
        return np.mean(aucs)

    def compute_best_oof(self):
        _all = []
        for k in range(self.num_oofs):
            auc = self.macro_multilabel_auc(
                self.y_true,
                self.all_oof_preds[
                    :, int(k * self.col_len) : int((k + 1) * self.col_len)
                ],
            )
            _all.append(auc)
            print("Model %i has OOF AUC = %.4f" % (k, auc))
        best_auc, best_oof_index = np.max(_all), np.argmax(_all)
        return best_auc, best_oof_index

    def forward_ensemble(self):
        DUPLICATES = False
        old_best_auc, best_oof_index = self.compute_best_oof()
        chosen_model = [best_oof_index]
        optimal_weights = []
        for oof_index in range(self.num_oofs):
            curr_model = self.all_oof_preds[
                :,
                int(best_oof_index * self.col_len) : int(
                    (best_oof_index + 1) * self.col_len
                ),
            ]
            for i, k in enumerate(chosen_model[1:]):
                # this step is confusing because it overwrites curr_model in the previous step. basically curr_model is reset to the best oof model initially, and then loop through to get the best oof
                curr_model = (
                    optimal_weights[i]
                    * self.all_oof_preds[
                        :, int(k * self.col_len) : int((k + 1) * self.col_len)
                    ]
                    + (1 - optimal_weights[i]) * curr_model
                )

            print("Searching for best model to add")

            # try add each model
            for i in range(self.num_oofs):
                print(i, ", ", end="")
                if not DUPLICATES and (i in chosen_model):
                    continue
                best_weight_index, best_score, patience_counter = 0, 0, 0
                for j in range(self.weight_interval):
                    temp = (j / self.weight_interval) * self.all_oof_preds[
                        :, int(i * self.col_len) : int((i + 1) * self.col_len)
                    ] + (1 - j / self.weight_interval) * curr_model
                    auc = self.macro_multilabel_auc(self.y_true, temp)

                    if auc > best_score:
                        best_score = auc
                        best_weight_index = j / self.weight_interval
                    else:
                        patience_counter += 1
                        # in this loop, if 10 increment in j does not lead to any increase in AUC, we break out
                    if patience_counter > self.patience:
                        break
                    if best_score > self.model_i_score:
                        self.model_i_score = best_score
                        self.model_i_index = i
                        self.model_i_weights = best_weight_index

            increment = self.model_i_score - old_best_auc
            if increment <= self.min_increase:
                print("No more significant increase")
                break
            # DISPLAY RESULTS
            print()
            print(
                "Ensemble AUC = %.4f after adding model %i with weight %.3f. Increase of %.4f"
                % (
                    self.model_i_score,
                    self.model_i_index,
                    self.model_i_weights,
                    increment,
                )
            )
            print()

            old_best_auc = self.model_i_score
            chosen_model.append(self.model_i_index)
            optimal_weights.append(self.model_i_weights)
            print(chosen_model)
        return chosen_model, optimal_weights


if __name__ == "__main__":
    PATH = "../input/ranzcr-oof-and-subs/"
    FILES = os.listdir(PATH)
    OOF = np.sort([f for f in FILES if "oof" in f])
    OOF_CSV = [pd.read_csv(PATH + k) for k in OOF]

    print("We have %i oof files..." % len(OOF))
    print()
    print(OOF)
    SUB = np.sort([f for f in FILES if "sub" in f])
    SUB_CSV = [pd.read_csv(PATH + k) for k in SUB]

    print("We have %i submission files..." % len(SUB))
    print()
    print(SUB)
    target_cols = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]

    pred_cols = [
        "ETT - Abnormal OOF",
        "ETT - Borderline OOF",
        "ETT - Normal OOF",
        "NGT - Abnormal OOF",
        "NGT - Borderline OOF",
        "NGT - Incompletely Imaged OOF",
        "NGT - Normal OOF",
        "CVC - Abnormal OOF",
        "CVC - Borderline OOF",
        "CVC - Normal OOF",
        "Swan Ganz Catheter Present OOF",
    ]
    forward_ens = ForwardEnsemble(
        dir=PATH,
        oof=OOF_CSV,
        weight_interval=200,
        patience=10,
        min_increase=0.0003,
        target_column_names=target_cols,
        pred_column_names=pred_cols,
    )
    m, w = forward_ens.forward_ensemble()

    y = np.zeros((len(SUB_CSV[0]), len(SUB) * len(pred_cols)))
    for k in range(len(SUB)):
        y[:, int(k * len(pred_cols)) : int((k + 1) * len(pred_cols))] = SUB_CSV[k][
            target_cols
        ].values

    md2 = y[:, int(m[0] * len(pred_cols)) : int((m[0] + 1) * len(pred_cols))]
    for i, k in enumerate(m[1:]):
        md2 = (
            w[i] * y[:, int(k * len(pred_cols)) : int((k + 1) * len(pred_cols))]
            + (1 - w[i]) * md2
        )
    plt.hist(md2, bins=100)
    plt.show()

    df = SUB_CSV[0].copy()
    df[target_cols] = md2
    df.to_csv("ensemble_sub.csv", index=False)
    df.head()

    ### If want to do col by col individually
    # for i, j in zip(target_cols, pred_cols):
    #     print(i, j)
    #     _target_cols = [i]
    #     _pred_cols = [j]
    #     forward_ens = ForwardEnsemble(
    #         dir=PATH,
    #         oof=OOF_CSV,
    #         weight_interval=200,
    #         patience=10,
    #         min_increase=0.0003,
    #         target_column_names=_target_cols,
    #         pred_column_names=_pred_cols,
    #     )
    #     m, w = forward_ens.forward_ensemble()

    #     y = np.zeros((len(SUB_CSV[0]), len(SUB) * len(_pred_cols)))
    #     for k in range(len(SUB)):
    #         y[:, int(k * len(_pred_cols)) : int((k + 1) * len(_pred_cols))] = SUB_CSV[
    #             k
    #         ][_target_cols].values

    #     md2 = y[:, int(m[0] * len(_pred_cols)) : int((m[0] + 1) * len(_pred_cols))]
    #     for i, k in enumerate(m[1:]):
    #         md2 = (
    #             w[i] * y[:, int(k * len(_pred_cols)) : int((k + 1) * len(_pred_cols))]
    #             + (1 - w[i]) * md2
    #         )
    #     plt.hist(md2, bins=100)
    #     plt.show()

    #     df = SUB_CSV[0].copy()
    #     df[_target_cols] = md2
    #     df.to_csv("ensemble_sub.csv", index=False)
    #     df.head()