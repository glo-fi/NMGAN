import torch


def truncated_normal(
        noise_size,
        threshold=1.0):
    samples = torch.randn(noise_size)
    smaller = samples < - threshold
    bigger = samples > threshold
    while smaller.sum() != 0 or bigger.sum() != 0:
        new_samples = torch.randn(noise_size)
        samples[smaller] = new_samples[smaller]
        samples[bigger] = new_samples[bigger]
        smaller = samples < - threshold
        bigger = samples > threshold
    return samples


def orthogonal_regularization(model, device):
    # (||W^T.W x (1 - I)||_F)^2
    penalty = torch.tensor(0.0, dtype=torch.float32, device=device)
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            shape = param.shape[0]
            flatten = param.view(shape, -1)
            beta_squared = torch.mm(flatten, flatten.t())  # W^T.W
            ones = torch.ones(shape, shape, dtype=torch.float32, device=device)
            diag = torch.eye(shape, dtype=torch.float32, device = device)
            penalty += ((beta_squared * (ones - diag)).to(device) ** 2).sum()
    return penalty
