import os

import torch
import numpy as np

from src.utils.config_parser import Config

dev = os.environ.get("device", "cpu")
device = torch.device(dev)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def greedy_loss(pred_feats, true_feats):
    loss_list = []
    count = 0
    for pf, tf in zip(pred_feats, true_feats):
        if len(pf) == 0 or len(tf) == 0:
            continue

        # abs_pred = torch.einsum("ij,ij->i", pf, pf)
        # abs_true = torch.einsum("kj,kj->k", tf, tf)
        # dot_pred_true = torch.einsum("ij,kj->ik", pf, tf)
        # tf = tf.to(pf.device)
        # pf = pf.to(tf.device)
        # print(count)
        # count += 1
        diff = tf - pf.unsqueeze(1)
        mse_loss = torch.einsum("ikj,ikj->ik", diff, diff) / diff.shape[2]

        # mse_loss = torch.mean(diff**2, dim=2)

        min_mse_values = torch.min(mse_loss, dim=1)[0]
        loss_list += [*min_mse_values]

    if len(loss_list) > 0:
        average_loss = torch.mean(torch.stack(loss_list), dim=0)
    else:
        average_loss = None
        # average_loss = torch.tensor(0, dtype=torch.float32, device=dev)
    return average_loss
