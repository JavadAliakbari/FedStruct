import os

import torch
import numpy as np

from src.utils.config_parser import Config

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def greedy_loss(pred_feats, true_feats, pred_missing):
    with torch.no_grad():
        pred_len = len(pred_feats)
        pred_missing_np = torch.round(pred_missing.detach()).type(torch.int32)
        pred_missing_np = torch.clip(pred_missing_np, 0, config.fedsage.num_pred)

        true_missing = np.array([len(true_feature) for true_feature in true_feats])

    loss_list = []
    for i in range(pred_len):
        num_missing = true_missing[i]
        if num_missing == 0 or pred_missing_np[i] == 0:
            continue

        pred = pred_feats[i][: pred_missing_np[i]]
        true_features = true_feats[i][:num_missing]

        # abs_pred = torch.einsum("ij,ij->i", pred, pred)
        # abs_true = torch.einsum("kj,kj->k", true_features, true_features)
        # dot_pred_true = torch.einsum("ij,kj->ik", pred, true_features)

        diff = true_features - pred.unsqueeze(1)
        mse_loss = torch.einsum("ikj,ikj->ik", diff, diff) / diff.shape[2]

        # mse_loss2 = torch.mean(diff**2, dim=2)

        min_mse_values = torch.min(mse_loss, dim=1)[0]
        loss_list += [*min_mse_values]

    if len(loss_list) > 0:
        average_loss = torch.mean(torch.stack(loss_list), dim=0)
    else:
        average_loss = torch.tensor([0], dtype=torch.float32)
    return average_loss
