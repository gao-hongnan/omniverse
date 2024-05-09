from typing import Any

import torch
import torchvision  # type: ignore[import-untyped]
from rich.pretty import pprint
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from tqdm import tqdm


def train(config: Any) -> None:
    """Run the training pipeline, however, the code below can be further
    modularized into functions for better readability and maintainability."""
    pprint(config)

    torch.manual_seed(config.train.seed)

    transform = transforms.Compose(
        [
            transforms.Resize((config.transform.image_size, config.transform.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.transform.mean, std=config.transform.std),
        ]
    )

    dataset = datasets.CIFAR10(root=config.datamodule.data_dir, train=True, transform=transform, download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.datamodule.batch_size,
        num_workers=config.datamodule.num_workers,
        shuffle=config.datamodule.shuffle,
    )

    model = getattr(torchvision.models, config.model.model_name)(pretrained=config.model.pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.model.num_classes)
    model = model.to(config.train.device)

    optimizer = getattr(torch.optim, config.optimizer.optimizer_name)(
        model.parameters(), **config.optimizer.optimizer_params
    )

    for epoch in range(config.train.num_epochs):
        model.train()
        with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config.train.num_epochs}]", unit="batch") as tepoch:
            for images, labels in tepoch:
                images = images.to(config.train.device)
                labels = labels.to(config.train.device)

                optimizer.zero_grad()
                logits = model(images)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

    model.eval()
    torch.save(model.state_dict(), config.stores.model_artifacts_dir / "model.pth")
