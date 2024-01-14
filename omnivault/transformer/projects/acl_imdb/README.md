# ACL IMDB

## Setup and Installation

We assume a macOS setup for this project. The steps for other operating systems
may vary.

### Step 5: Download and Set Up the ACL IMDB Dataset

For training models on sentiment analysis, follow these steps to download and
set up the ACL IMDB dataset:

1. **Create the Dataset Directory** for the ACL IMDB dataset:

    ```bash
    (venv) $ mkdir -p ./data/aclImdb
    ```

2. **Download and Extract the Dataset**:

    ```bash
    (venv) $ curl -o ./data/aclImdb/aclImdb_v1.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    (venv) $ tar -xf ./data/aclImdb/aclImdb_v1.tar.gz -C ./data/aclImdb && rm ./data/aclImdb/aclImdb_v1.tar.gz
    ```

This will download the ACL IMDB dataset into `./data/aclImdb` and extract its
contents there.
