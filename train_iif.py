import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from utils.image import *
from utils.io import *
from modules.conv2d_resblock import Conv2DResBlock


class IIF(nn.Module):
    def __init__(self, dims=3, channels=128, blocks=8):
        super().__init__()

        self.dims = dims

        self.encoder = nn.Sequential(
            nn.Conv2d(self.dims, channels, 9, 1, 4),
            Conv2DResBlock(channels, channels, 5, 1, 2),
            Conv2DResBlock(channels, channels, 5, 1, 2),
            Conv2DResBlock(channels, channels, 5, 1, 2),
            Conv2DResBlock(channels, channels, 5, 1, 2),
            nn.Conv2d(channels, self.dims, 1, 1, 0),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 + 32 * 32 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def _sample_latent(self, Z, x):
        z = Z
        return z

    def forward(self, image, mask, x):
        Z = self.encoder(image.unsqueeze(0))
        Z *= mask
        z = self._sample_latent(Z, x)
        # Z: 3, 32, 32
        # x: n, 2

        z = z.flatten()
        # z: 3072

        decoder_input = torch.hstack((x, z.unsqueeze(0).repeat(x.shape[0], 1)))
        s = self.decoder(decoder_input)

        return s


def from_coords(shape, x, s):
    image = torch.zeros(shape, device=x.device)
    image[:, x.T[0], x.T[1]] = s.T

    return image


def to_coords(image: torch.Tensor, mask: torch.Tensor):
    coords = mask[0].nonzero(as_tuple=True)
    return image[:, coords[0], coords[1]].T, mask[0].nonzero()


def train(model, dataloader, optimizer, device):
    model.train()
    netloss = 0

    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)[0]
        masked, mask = mask_random(images, 0.5)
        
        s, x = to_coords(images, mask.logical_not())

        # s: (n, 3)
        # x: (n, 2)

        optimizer.zero_grad()
        s_pred = model(masked, mask, x)

        loss = ((s - s_pred) ** 2).mean()
        loss.backward()
        optimizer.step()

        netloss += loss

        if i % 10000 == 0:
            print(f"iteration {i} - loss {loss}")

    return netloss.item() / len(dataloader)


def validate(model, dataiter, device):
    model.eval()
    with torch.no_grad():
        images, _ = next(dataiter)
        images = images.to(device)[0]

        masked, mask = mask_random(images, 0.5)
        
        s, x = to_coords(images, mask.logical_not())

        s_pred = model(masked, mask, x)
        loss = ((s - s_pred) ** 2).mean()

    return loss.item()


if __name__ == "__main__":
    load_prev = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "iif.tar"

    trainset = CelebA(
        "data",
        "train",
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((32, 32))]
        ),
    )
    trainloader = DataLoader(
        trainset,
        1,
        shuffle=True,
    )

    validset = CelebA(
        "data",
        "valid",
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((32, 32))]
        ),
    )
    validloader = DataLoader(validset, 1, shuffle=True)

    model = IIF().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if load_prev:
        epoch, model, optimizer, history = load(
            checkpoint_path, model, optimizer, load_history=True
        )
        epoch += 1
    else:
        epoch = 0
        history = {"trainlosses": [], "validlosses": []}

    epochs = 100
    for epoch in range(epoch, epoch + epochs):
        try:
            print(f"epoch {epoch}")
            trainloss = train(model, trainloader, optimizer, device)
            validloss = validate(model, iter(validloader), device)

            history["trainlosses"].append(trainloss)
            history["validlosses"].append(validloss)

            save(checkpoint_path, epoch, model, optimizer, history)

        except KeyboardInterrupt:
            break
