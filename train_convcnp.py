import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from utils.image import *
from models import Conv2dCNP


def train(model, dataloader, optimizer, device):
    model.train()
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        loss_fn = lambda x: -x.log_prob(images).sum(1).mean()

        erased, mask = mask_random(images, 0.5)
        erased, mask = erased.to(device), mask.to(device)

        optimizer.zero_grad()
        p = model(erased, mask)

        loss = loss_fn(p)
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"iteration {i} - loss {loss}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "convcnp.pth"

    trainset = CIFAR10("data", "train", download=False, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, 16, True)

    model = Conv2dCNP(3, 128, 9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 20

    try:
        for i in range(epochs):
            print(f"epoch {i}")
            train(model, trainloader, optimizer, device)
    except KeyboardInterrupt:
        pass

    torch.save(model.state_dict(), weights_path)
