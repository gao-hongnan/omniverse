import importlib
from types import ModuleType
from typing import Dict, Generic, Tuple, Type

from pydantic import BaseModel, Field

from omnivault._types._generic import DynamicClass, T, V


class DynamicClassFactory(BaseModel, Generic[DynamicClass]):
    r"""
    A factory class for dynamic instantiation of classes based on a configuration.

    Using a `TypeVar` for the generic type `DynamicClass` is better than using `Any`
    since we really have no way to know the return type of `create_instance()`.

    This class serves as a base for configuration management, particularly useful
    in scenarios involving complex systems or large libraries where classes or
    functions come with numerous arguments. It centralizes the management of
    these configurations, simplifying instantiation and enhancing flexibility and
    maintainability of the code.

    Attributes
    ----------
    name : str
        Fully qualified class name to be instantiated. This should include
        both the module and the class name (e.g., 'mymodule.MyClass'). The factory
        dynamically imports the module, accesses the class, and creates an
        instance using other configuration parameters provided in the subclass.

    Methods
    -------
    pop_name_and_return_remaining_kwargs() -> KwargsDict:
        Extracts the configuration parameters excluding 'name' attribute.

    create_instance() -> Any:
        Dynamically creates an instance of the class specified in 'name' attribute
        using the provided configuration parameters.

    Note
    ----
    - This class is particularly beneficial for larger, complex systems where
      configurations need to be changed frequently or shared across components.
    - It is designed to be extended by specific configuration classes tailored to
      particular use cases.

    Examples
    --------
    >>> class TextSplitterConfig(DynamicClassFactory):
    ...     name: str = "langchain.text_splitter.RecursiveCharacterTextSplitter"
    ...     separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    ...     chunk_size: int = 300
    ...     chunk_overlap: int = 50
    ...     length_function: Callable = Field(default_factory=lambda: len)
    >>> splitter_config = TextSplitterConfig()
    >>> text_splitter = splitter_config.create_instance()
    """

    name: str = Field(
        ...,
        description=(
            "The 'name' field should contain the fully qualified class name "
            "that this factory will instantiate. This includes both the module "
            "and the class name. For example, 'mymodule.MyClass'. The factory "
            "will dynamically import the specified module, access the class, "
            "and create an instance of it using the other configuration "
            "parameters provided in the subclass."
        ),
    )

    def pop_name_and_return_remaining_kwargs(self) -> Dict[str, V]:
        return {key: value for key, value in self.model_dump(mode="python").items() if key != "name"}

    def split_class_name(self) -> Tuple[str, str]:
        """Split the full class name into module and class names."""
        # pylint: disable=no-member
        if not self.name or self.name.startswith(".") or self.name.endswith("."):
            raise ValueError(f"Invalid class name format: {self.name}")

        try:
            module_name, class_name = self.name.rsplit(".", 1)
            return module_name, class_name
        except ValueError as err:
            raise ValueError(f"Invalid class name format: {self.name}") from err

    def import_module(self, module_name: str) -> ModuleType:
        """Import and return the module based on the module name."""
        try:
            return importlib.import_module(module_name)
        except ImportError as err:
            raise ImportError(f"Module {module_name} cannot be imported") from err

    def get_class_from_module(self, module: ModuleType, class_name: str) -> Type[DynamicClass]:
        """Retrieve the class from the imported module."""
        try:
            return getattr(module, class_name)  # type: ignore[no-any-return]
        except AttributeError as err:
            raise AttributeError(f"Class {class_name} not found in module {module.__name__}") from err

    def create_instance(self, *args: T, **kwargs: T) -> DynamicClass:
        """
        Note
        ----
        1.  self.name is an instance of FieldInfo but at run time it should be
            str, so I disable pylint complaints.
        2.  args and kwargs are passed to the class constructor to handle cases
            such as PyTorch Optimizer that requires parameters to be passed in
            the constructor.
        """
        # fmt: off
        module_name, class_name = self.split_class_name()
        module                  = self.import_module(module_name)
        class_                  = self.get_class_from_module(module, class_name)
        # fmt: on
        return class_(*args, **self.pop_name_and_return_remaining_kwargs(), **kwargs)
