import torch

def save(path, epoch, model, optimizer):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    
    print(f"saved checkpoint at epoch {epoch} to {path}")

def load(path, model, optimizer=None):
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    
    print(f"loaded checkpoint at epoch {epoch} from {path}")
    
    return epoch, model, optimizer
