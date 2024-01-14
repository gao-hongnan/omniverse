from typing import Any, Dict, List, Type, Union

from pydantic import BaseModel, Field, field_validator

from omnivault._types._alias import NotGiven


class DataConfig(BaseModel):
    """The data config."""

    context_length: int = Field(
        default=128,
        description="The context length depends on how we tokenize, whether on a character level or word level.",
    )
    dataset_name: str = Field(
        default=None,
        description="The name of the dataset. Also the stem of the url or data path, for example, if the filepath is `data/abc.txt`, then the dataset name is `abc`.",
    )
    dataset_size: Union[int, None] = Field(default=2, description="The size of the dataset.")
    dataset_path: str = Field(default=None, description="The path to the dataset.")
    dataset_dir: str = Field(default=None, description="The directory to the dataset.")
    dataset_url: str = Field(default=None, description="The url to the dataset.")

    split: Union[List[float], None] = Field(default=[0.7, 0.1, 0.2], description="The split ratio of the dataset.")

    collate_fn: Union[Dict[str, Any], NotGiven, None] = Field(
        default={
            "batch_first": True,
            "pad_token_id": 16,
        },  # TODO: `pad_token_id` should be interpolated from `MaybeConstant`.
        description="The collate function config.",
    )

    train_loader: Dict[str, Any] = Field(
        default={"batch_size": 32, "shuffle": True, "num_workers": 0, "pin_memory": False, "drop_last": False},
        description="The train loader config.",
    )
    valid_loader: Union[Dict[str, Any], None] = Field(
        default={"batch_size": 32, "shuffle": False, "num_workers": 0, "pin_memory": False, "drop_last": False},
        description="The validation loader config.",
    )
    test_loader: Union[Dict[str, Any], None] = Field(
        default={"batch_size": 32, "shuffle": False, "num_workers": 0, "pin_memory": False, "drop_last": False},
        description="The test loader config.",
    )

    class Config:
        arbitrary_types_allowed = True

    # FIXME: hard to handle since collate_fn can be NotGiven but yaml config can only indicate null.
    # unless I do __target__?
    @field_validator("collate_fn")
    def coerce_collate_fn(cls: Type["DataConfig"], v: Union[Dict[str, Any], NotGiven, None]) -> Dict[str, Any]:  # type: ignore
        if v is None:
            return {}
        if isinstance(v, NotGiven):
            return {}
        else:
            return v
