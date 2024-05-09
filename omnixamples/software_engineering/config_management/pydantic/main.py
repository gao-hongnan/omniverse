from pathlib import Path

from omnixamples.software_engineering.config_management.pydantic.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    StoresConfig,
    TrainConfig,
    TransformConfig,
)
from omnixamples.software_engineering.config_management.train import train

if __name__ == "__main__":
    config = Config(
        model=ModelConfig(
            model_name="resnet18",
            pretrained=True,
            in_chans=3,
            num_classes=10,
            global_pool="avg",
        ),
        transform=TransformConfig(
            image_size=256,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        datamodule=DataConfig(data_dir=Path("./data"), batch_size=32, num_workers=0),
        optimizer=OptimizerConfig(
            optimizer_name="AdamW",
            optimizer_params={
                "lr": 0.0003,
                "betas": [0.9, 0.999],
                "amsgrad": False,
                "weight_decay": 1e-06,
                "eps": 1e-08,
            },
        ),
        stores=StoresConfig(
            project_name="cifar10",
            unique_id="12345",
            logs_dir=Path("./logs"),
            model_artifacts_dir=Path("./artifacts"),
        ),
        train=TrainConfig(device="cpu", project_name="cifar10", debug=True, seed=1992, num_epochs=3, num_classes=10),
    )

    train(config)
