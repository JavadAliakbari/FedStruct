import torch
import torch.nn.functional as F

from src.utils import config


def greedy_loss(pred_feats, true_feats, pred_missing):
    if config.cuda:
        pred_missing = pred_missing.cpu()
    total_loss = torch.tensor([0.0], requires_grad=True)
    if config.cuda:
        loss = loss.cuda()
    pred_len = len(pred_feats)
    pred_missing_np = torch.round(pred_missing.detach()).type(torch.int32)
    pred_missing_np = torch.clip(pred_missing_np, 0, config.num_pred)
    counter = torch.tensor([0], dtype=torch.float32, requires_grad=True)
    for i in range(pred_len):
        for pred_j in range(pred_missing_np[i]):
            num_missing = len(true_feats[i])
            loss = torch.tensor([10000.0], requires_grad=True)
            for true_k in range(num_missing):
                temp = F.mse_loss(
                    pred_feats[i][pred_j].unsqueeze(0).float(),
                    true_feats[i][true_k].unsqueeze(0).float(),
                ).squeeze(0)
                if torch.sum(temp) < torch.sum(loss.data):
                    loss = temp

            if num_missing > 0:
                total_loss = total_loss + loss
                counter = counter + 1

    if counter > 0:
        average_loss = total_loss / counter
    else:
        average_loss = total_loss
    return average_loss
