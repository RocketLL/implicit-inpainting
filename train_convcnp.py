import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from utils.image import *
from utils.io import *
from models import Conv2dCNP


def train(model, dataloader, optimizer, device):
    model.train()
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        loss_fn = lambda x: -x.log_prob(images).sum(1).mean()

        erased, mask = mask_random(images, torch.rand(1) * 0.8 + 0.1)

        optimizer.zero_grad()
        p = model(erased, mask)

        loss = loss_fn(p)
        loss.backward()
        optimizer.step()
        
        if i % 500 == 0:
            print(f"iteration {i} - loss {loss}")
        
    return loss
            

if __name__ == "__main__":
    load_prev = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "convcnp.tar"
    
    trainset = CelebA("data", "train", download=False, transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]))
    trainloader = DataLoader(trainset, 32, True)

    model = Conv2dCNP(3, 128, 8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if load_prev:
        epoch, model, optimizer = load(checkpoint_path, model, optimizer)
    else:
        epoch = 0
    
    epochs = 100
    for epoch in range(epoch, epoch + epochs):
        try:
            print(f"epoch {epoch}")
            loss = train(model, trainloader, optimizer, device)
        except KeyboardInterrupt:
            break
            
        save(checkpoint_path, epoch, model, optimizer)
        
    save(checkpoint_path, epoch, model, optimizer)
