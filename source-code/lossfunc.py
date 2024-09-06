import torch


def dist_loss(data, min_dist, max_dist=20):
    pairwise_dist = torch.cdist(data, data)
    dist = pairwise_dist - min_dist
    bigdist = max_dist - pairwise_dist
    loss = torch.exp(-dist) + torch.exp(-bigdist)
    return loss


def cdisttf(data_1, data_2):
    prod = torch.sum(
        (data_1.unsqueeze(1) - data_2.unsqueeze(0)) ** 2, dim=2
    )
    return (prod + 1e-10).sqrt()