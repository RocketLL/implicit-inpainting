import torch
from torchvision import transforms
from torchvision.transforms import functional


def mask_rectangle(image: torch.Tensor, scale, ratio):
    params = transforms.RandomErasing.get_params(image, scale, ratio, torch.zeros(1))

    erased = functional.erase(image, *params)
    mask = functional.erase(torch.ones(image.shape), *params)

    return erased, mask


def mask_superresolution(image: torch.Tensor, n=2):
    w = image.shape[-2] * n
    h = image.shape[-1] * n

    erased = torch.zeros(
        (*image.shape[0:-2], w, h), dtype=image.dtype, device=image.device
    )
    erased[..., :, ::n, ::n] = image

    mask = torch.zeros_like(erased)
    mask[..., :, ::n, ::n] = 1

    return erased, mask


def mask_random(image: torch.Tensor, p=0.5):
    mask = torch.zeros(image.shape[-3:])

    mask[0, :, :].bernoulli_(p)
    mask[:, :, :] = mask[0, :, :].unsqueeze(-3)

    return image * mask, mask.broadcast_to(image.shape)
