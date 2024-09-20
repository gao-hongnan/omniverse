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
