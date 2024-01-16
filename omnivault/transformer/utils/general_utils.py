import os
import shutil
import tempfile
from typing import Generator

import requests


def download_and_read_sequences(url: str, dataset_name: str) -> Generator[str, None, None]:
    temp_dir = tempfile.mkdtemp()

    try:
        response = requests.get(url)
        response.raise_for_status()

        temp_file_path = os.path.join(temp_dir, f"{dataset_name}.txt")
        with open(temp_file_path, "wb") as file:
            file.write(response.content)

        with open(temp_file_path, "r") as file:
            for line in file:
                yield line.strip()

    finally:
        shutil.rmtree(temp_dir)
