import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from utils.image import *
from utils.io import *
from models import Conv2dCNP
import math


def log_prob(loc, scale, value):
    var = scale ** 2
    log_scale = scale.log()
    return (
        -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    )


def train(model, dataloader, optimizer, device):
    model.train()
    netloss = 0

    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)

        erased, mask = mask_random(images, torch.rand(1) * 0.8 + 0.1)

        optimizer.zero_grad()
        mean, std = model(erased, mask)

        loss = -log_prob(mean, std, images).sum(1).mean()
        loss.backward()
        optimizer.step()

        netloss += loss.item()

        if i % 500 == 0:
            print(f"iteration {i} - loss {loss.item()}")

    return netloss / len(dataloader)


def validate(model, dataiter, device):
    model.eval()
    with torch.no_grad():
        images, _ = next(dataiter)
        images = images.to(device)

        erased, mask = mask_random(images, 0.5)

        mean, std = model(erased, mask)
        loss = -log_prob(mean, std, images).sum(1).mean()
        mse = ((mean - images) ** 2).mean().item()

    return loss.item(), mse


if __name__ == "__main__":
    load_prev = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "convcnp.tar"

    trainset = CelebA(
        "data",
        "train",
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((64, 64))]
        ),
    )
    trainloader = DataLoader(
        trainset,
        64,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
    )

    validset = CelebA(
        "data",
        "valid",
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((64, 64))]
        ),
    )
    validloader = DataLoader(validset, 8, shuffle=True)

    model = Conv2dCNP(3, 128, 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if load_prev:
        epoch, model, optimizer, history = load(
            checkpoint_path, model, optimizer, load_history=True
        )
        epoch += 1
    else:
        epoch = 0
        history = {"trainlosses": [], "validlosses": [], "validmses": []}

    epochs = 100
    for epoch in range(epoch, epoch + epochs):
        try:
            print(f"epoch {epoch}")
            trainloss = train(model, trainloader, optimizer, device)
            validloss, validmse = validate(model, iter(validloader), device)

            history["trainlosses"].append(trainloss)
            history["validlosses"].append(validloss)
            history["validmses"].append(validmse)

            save(checkpoint_path, epoch, model, optimizer, history)

        except KeyboardInterrupt:
            break
