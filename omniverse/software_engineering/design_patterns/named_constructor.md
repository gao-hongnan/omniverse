---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Named Constructor

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)
[![Code](https://img.shields.io/badge/View-Code-blue?style=flat-square&logo=github)](https://github.com/gao-hongnan/omniverse/blob/288357646b8e4043d0b0a81c8b6b6c600fbd2efd/omnixamples/software_engineering/design_patterns/named_constructor/from_classmethod.py)

```{contents}
```

The `from_xxx` pattern is increasingly popular in modern Python libraries. I did
not realise that it is actually called the **Named Constructor
Pattern/Alternative Constructor Pattern**. And it is also considered sometimes
an
[anti-pattern](https://softwareengineering.stackexchange.com/questions/358502/why-named-constructors-are-getting-popular-shouldnt-be-an-antipattern).
Nevertheless, it is a pretty neat pattern and I use it a lot in my codebase.
Let's consider a simple example below where we have a configuration management
system for a web application.

1. Multiple Configuration Classes:

    - `DatabaseConfig`: Holds database-specific settings.
    - `CacheConfig`: Stores caching configuration.
    - `AppConfig`: The main configuration class that incorporates both database
      and cache configs, along with other application settings.

2. Alternative Constructors:

    - `from_yaml`: Creates an `AppConfig` instance from a YAML file.
    - `from_json`: Creates an `AppConfig` instance from a JSON file.
    - `from_env`: Creates an `AppConfig` instance from environment variables.
    - `from_dict`: A method that centralizes the logic for creating an
      `AppConfig` instance from a dictionary. This is used by all alternative
      constructors, promoting code reuse.

3. Flexibility:

    - The system can handle different configuration sources (YAML, JSON,
      environment variables) without changing the core `AppConfig` structure.
    - Additional settings can be included, allowing for extensibility.

This pattern is particularly useful in this scenario because:

1. It provides a clean, consistent interface for creating `AppConfig` objects
   from various sources.
2. It encapsulates the complexity of parsing different file formats and
   environment variables.
3. It allows for easy extension to support additional configuration sources in
   the future.

```{code-cell} ipython3
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Type

import yaml
from pydantic import BaseModel
from rich.pretty import pprint


class DatabaseConfig(BaseModel):
    host: str
    port: int
    username: str
    password: str
    db_name: str


class CacheConfig(BaseModel):
    cache_type: str


class AppConfig(BaseModel):
    app_name: str
    environment: str
    debug: bool
    database: DatabaseConfig
    cache: CacheConfig

    @classmethod
    def from_yaml(cls: Type[AppConfig], file_path: str) -> AppConfig:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls.from_dict(config_data)

    @classmethod
    def from_json(cls: Type[AppConfig], file_path: str) -> AppConfig:
        with open(file_path, "r") as file:
            config_data = json.load(file)
        return cls.from_dict(config_data)

    @classmethod
    def from_env(cls: Type[AppConfig]) -> AppConfig:
        config_data = {
            "app_name": os.getenv("APP_NAME"),
            "environment": os.getenv("ENVIRONMENT"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "database": {
                "host": os.getenv("DB_HOST"),
                "port": int(os.getenv("DB_PORT", 5432)),
                "username": os.getenv("DB_USERNAME"),
                "password": os.getenv("DB_PASSWORD"),
                "db_name": os.getenv("DB_NAME"),
            },
            "cache": {
                "cache_type": os.getenv("CACHE_TYPE"),
                "host": os.getenv("CACHE_HOST"),
                "port": int(os.getenv("CACHE_PORT", 6379)),
            },
        }
        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls: Type[AppConfig], config_data: Dict[str, Any]) -> AppConfig:
        db_config = DatabaseConfig(**config_data["database"])
        cache_config = CacheConfig(**config_data["cache"])
        return cls(
            app_name=config_data["app_name"],
            environment=config_data["environment"],
            debug=config_data["debug"],
            database=db_config,
            cache=cache_config,
        )


if __name__ == "__main__":
    yaml_string = """
    app_name: MyApp
    environment: development
    debug: true
    database:
      host: localhost
      port: 5432
      username: user
      password: pass
      db_name: mydb
    cache:
      cache_type: redis
      host: localhost
      port: 6379
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_file:
        temp_file.write(yaml_string)
        temp_file_path = temp_file.name

    try:
        yaml_config = AppConfig.from_yaml(temp_file_path)
        pprint(yaml_config)
    finally:
        os.unlink(temp_file_path)

    # From JSON file
    json_string = """
    {
        "app_name": "MyApp",
        "environment": "development",
        "debug": true,
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "user",
            "password": "pass",
            "db_name": "mydb"
        },
        "cache": {
            "cache_type": "redis",
            "host": "localhost",
            "port": 6379
        }
    }
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
        temp_file.write(json_string)
        temp_file_path = temp_file.name

    try:
        json_config = AppConfig.from_json(temp_file_path)
        pprint(json_config)
    finally:
        os.unlink(temp_file_path)
```

## References And Further Readings

-   [Named Constructor Pattern](https://softwareengineering.stackexchange.com/questions/358502/why-named-constructors-are-getting-popular-shouldnt-be-an-antipattern)
-   [Alternative Constructor Pattern](https://stackoverflow.com/questions/73569542/how-class-methods-can-be-alternative-constructors-as-they-just-return-us-objects)
