import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from utils.image import *
from utils.io import *
from modules.conv2d_resblock import Conv2DResBlock

BATCH_SIZE = 16
W = H = 32


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
            nn.Linear(2 + W * H * 3, 256),
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
        # image: (BATCH_SIZE, 3, W, H)
        # mask:  (BATCH_SIZE, 3, W, H)
        # x:     (BATCH_SIZE, n, 2)

        Z = self.encoder(image)
        Z *= mask
        z = self._sample_latent(Z, x)

        # Z: (BATCH_SIZE, 3, W, H)
        # x: (BATCH_SIZE, n, 2)

        z = z.flatten(1)
        # z: (BATCH_SIZE, 3 * W * H)

        decoder_input = torch.cat((x, z.unsqueeze(1).repeat(1, x.shape[-2], 1)), -1)
        s = self.decoder(decoder_input)

        return s


def to_coords(image: torch.Tensor, mask: torch.Tensor):
    # image: (BATCH_SIZE, 3, W, H)
    # mask: (BATCH_SIZE, 3, W, H)
    coords = mask[0, 0].nonzero(as_tuple=True)
    image_data = image[..., :, coords[0], coords[1]].transpose(-1, -2)

    coords = mask[0, 0].nonzero().unsqueeze(0).repeat(image_data.shape[0], 1, 1)

    return (image_data, coords)


def from_coords(shape, x, s):
    # x: (BATCH_SIZE, n, 2)
    # s: (BATCH_SIZE, n, 3)
    image = torch.zeros(shape, device=x.device)
    # image: (BATCH_SIZE, n, W, H)

    for i in range(shape[0]):
        image[i, :, x.T[0], x.T[1]] = s.T

    return image


def normalize_coords(x):
    x[..., 0] = x[..., 0] / (W - 1) * 2 - 1.0
    x[..., 1] = x[..., 1] / (H - 1) * 2 - 1.0
    return x


def train(model, dataloader, optimizer, device):
    model.train()
    netloss = 0

    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        masked, mask = mask_random(
            images, torch.rand(1) * 0.8 + 0.1
        )  # uniform(0.1, 0.9)

        s, x = to_coords(images, mask.logical_not())
        x = normalize_coords(x)
        # s: (BATCH_SIZE, n, 3)
        # x: (BATCH_SIZE, n, 2)

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
        images = images.to(device)

        masked, mask = mask_random(images, 0.5)

        s, x = to_coords(images, mask.logical_not())
        x = normalize_coords(x)

        s_pred = model(masked, mask, x)
        loss = ((s - s_pred) ** 2).mean()

    return loss.item()


if __name__ == "__main__":
    load_prev = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "iif.tar"

    trainset = CelebA(
        "data",
        "train",
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((W, H))]
        ),
    )
    trainloader = DataLoader(
        trainset,
        BATCH_SIZE,
        shuffle=True,
    )

    validset = CelebA(
        "data",
        "valid",
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((W, H))]
        ),
    )
    validloader = DataLoader(validset, BATCH_SIZE, shuffle=True)

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
