import torch


def greedy_loss(pred_feats, true_feats):
    loss_list = []
    for pf, tf in zip(pred_feats, true_feats):
        if len(pf) == 0 or len(tf) == 0:
            continue

        diff = tf - pf.unsqueeze(1)
        mse_loss = torch.einsum("ikj,ikj->ik", diff, diff) / diff.shape[2]

        min_mse_values = torch.min(mse_loss, dim=1)[0]
        loss_list += [*min_mse_values]

    if len(loss_list) > 0:
        average_loss = torch.mean(torch.stack(loss_list), dim=0)
    else:
        average_loss = None
    return average_loss
