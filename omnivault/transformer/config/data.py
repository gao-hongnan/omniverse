from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnivault._types._alias import NotGiven


class DataConfig(BaseModel):
    """The data config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context_length: Annotated[
        int,
        Field(description="The context length depends on how we tokenize, whether on a character level or word level."),
    ] = 128
    dataset_name: Annotated[
        str | None,
        Field(
            description="The name of the dataset. Also the stem of the url or data path, for example, if the filepath is `data/abc.txt`, then the dataset name is `abc`."
        ),
    ] = None
    dataset_size: Annotated[int | None, Field(description="The size of the dataset.")] = 2
    dataset_path: Annotated[str | None, Field(description="The path to the dataset.")] = None
    dataset_dir: Annotated[str | None, Field(description="The directory to the dataset.")] = None
    dataset_url: Annotated[str | None, Field(description="The url to the dataset.")] = None

    split: list[float] | None = Field(
        default_factory=lambda: [0.7, 0.1, 0.2], description="The split ratio of the dataset."
    )

    collate_fn: dict[str, Any] | NotGiven | None = Field(
        default_factory=lambda: {
            "batch_first": True,
            "pad_token_id": 16,
        },  # TODO: `pad_token_id` should be interpolated from `MaybeConstant`.
        description="The collate function config.",
    )

    train_loader: dict[str, Any] = Field(
        default_factory=lambda: {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        },
        description="The train loader config.",
    )
    valid_loader: dict[str, Any] | None = Field(
        default_factory=lambda: {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        },
        description="The validation loader config.",
    )
    test_loader: dict[str, Any] | None = Field(
        default_factory=lambda: {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        },
        description="The test loader config.",
    )

    # FIXME: hard to handle since collate_fn can be NotGiven but yaml config can only indicate null.
    # unless I do __target__?
    @field_validator("collate_fn")
    @classmethod
    def coerce_collate_fn(cls: type[Self], v: dict[str, Any] | NotGiven | None) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, NotGiven):
            return {}
        else:
            return v
