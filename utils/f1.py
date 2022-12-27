from sklearn.metrics import f1_score
import torch

def compute_F1(gt, pred):
    f1 = f1_score(torch.ravel(gt).cpu().detach().numpy(), torch.ravel(pred).cpu().detach().numpy(), zero_division = 0)
    return f1