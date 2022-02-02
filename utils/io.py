import torch

def save(path, epoch, model, optimizer, history=None):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }, path)
    
    print(f"saved checkpoint at epoch {epoch} to {path}")

def load(path, model, optimizer=None, load_history=False):
    checkpoint = torch.load(path)
    history = None
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    
    if load_history:
        history = checkpoint["history"]
    
    print(f"loaded checkpoint at epoch {epoch} from {path}")
    
    return epoch, model, optimizer, history
