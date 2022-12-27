import torch
import os

def save_checkpoint(root, model1,model2, better):
    if better:
        fpath1 = os.path.join(root, 'best_checkpoint_model1.pth')
        fpath2 = os.path.join(root, 'best_checkpoint_model2.pth')
    else:
        fpath1 = os.path.join(root, 'last_checkpoint_model1.pth')
        fpath2 = os.path.join(root, 'last_checkpoint_model2.pth')
    torch.save(model1.state_dict(), fpath1)
    torch.save(model2.state_dict(), fpath2)