import torch
import numpy as np
import torch.nn.functional as F

from src.utils.config_parser import Config

config = Config()
config.num_pred = 5


def greedy_loss(pred_feats, true_feats, pred_missing):
    # if config.cuda:
    # pred_missing = pred_missing.cpu()
    # total_loss = torch.tensor([0.0], requires_grad=True)
    # if config.cuda:
    #     loss = loss.cuda()
    with torch.no_grad():
        pred_len = len(pred_feats)
        pred_missing_np = torch.round(pred_missing.detach()).type(torch.int32)
        pred_missing_np = torch.clip(pred_missing_np, 0, config.num_pred)

        true_missing = np.array([len(true_feature) for true_feature in true_feats])
        array_size = sum(pred_missing_np[true_missing > 0])
        ind = 0

    loss_list = torch.zeros(array_size, dtype=torch.float32)
    # counter = torch.tensor([0], dtype=torch.float32, requires_grad=True)
    for i in range(pred_len):
        for pred_j in range(pred_missing_np[i]):
            with torch.no_grad():
                num_missing = true_missing[i]
            loss = torch.tensor([10000.0])
            for true_k in range(num_missing):
                temp = F.mse_loss(
                    pred_feats[i][pred_j],
                    true_feats[i][true_k],
                )
                if temp < loss:
                    loss = temp

            if num_missing > 0:
                loss_list[ind] = loss
                ind += 1
                # total_loss = total_loss + loss
                # counter = counter + 1

    if loss_list.shape[0] > 0:
        average_loss = loss_list.mean()
    else:
        average_loss = torch.tensor([0], dtype=torch.float32)
    return average_loss
