# Application 1: Predicting Iris Species and Deploying with Docker using Streamlit (Revisit my common-utils)

```{contents}
:local:
```

In this section, we will use the
[**Iris Dataset**](https://en.wikipedia.org/wiki/Iris_flower_data_set) to make a
simple machine learning model to predict the species of an iris flower given the
sepal length, sepal width, petal length, and petal width.

We will then containerize the model using Docker and deploy it using
[**Streamlit**](https://streamlit.io/).

The code is self contained below:

```python title="app.py" linenums="1"
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def download_iris_data(url: str, filename: str, data_path: str = "./data") -> None:
    """
    This function downloads the iris dataset from the provided URL.
    """
    Path(data_path).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, Path(data_path) / filename)


def load_and_preprocess_data(filename: str, data_path: str = "./data") -> tuple:
    """
    This function loads and preprocesses the data.
    It returns the features X, labels y and the label encoder.
    """
    filepath = f"{data_path}/{filename}"
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    # Load the data with the column names
    df = pd.read_csv(filepath, names=column_names)

    # Prepare data
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])

    # Split the data into features and labels
    X = df.drop("species", axis=1)
    y = df["species"]

    return X, y, le


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    This function trains a random forest classifier and returns the trained model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model


def get_user_input() -> pd.DataFrame:
    """
    This function gets the input features from the user and returns them as a dataframe.
    """
    # Get input features from the user
    sepal_length = st.number_input("Sepal Length", 0.0, 10.0, step=0.1)
    sepal_width = st.number_input("Sepal Width", 0.0, 10.0, step=0.1)
    petal_length = st.number_input("Petal Length", 0.0, 10.0, step=0.1)
    petal_width = st.number_input("Petal Width", 0.0, 10.0, step=0.1)

    # Convert the user input into a dataframe
    input_data = pd.DataFrame(
        {
            "sepal_length": [sepal_length],
            "sepal_width": [sepal_width],
            "petal_length": [petal_length],
            "petal_width": [petal_width],
        }
    )

    return input_data


# URL of the iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
filename = "iris.csv"
data_path = "./data"

download_iris_data(url, filename, data_path=data_path)

# Load and preprocess the data
X, y, le = load_and_preprocess_data(filename, data_path=data_path)

# Train the model
model = train_model(X, y)

# Create the Streamlit app
st.title("Iris Species Prediction")
st.write("Enter the flower's measurements to predict the species.")

# Get the user input
input_data = get_user_input()

# Make a prediction and show the result
if st.button("Predict"):
    result = model.predict(input_data)
    species = le.inverse_transform(result)[0]
    st.success(f"The species of the flower is most likely {species}.")
```

and the corresponding `requirements.txt` file:

```text title="requirements.txt"
numpy
pandas
scikit-learn
streamlit
```

Finally, the corresponding `Dockerfile` is:

```Dockerfile title="Iris Dockerfile" linenums="1"
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Run pip to install dependencies as root
RUN python3 -m pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application
COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create the data directory in the container
RUN mkdir -p /app/data

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
```

Note in particular I `COPY` twice in the `Dockerfile` to take advantage of
Docker's caching mechanism. The first `COPY` is for the `requirements.txt` file
and the second `COPY` is for the rest of the application.

This is because we often change the code in the application but rarely change
the dependencies. Docker builds by layers in the `Dockerfile` and if we place
the `COPY` for the whole `app` directory before the `RUN pip install` command,
then every time we change the code, Docker will have to reinstall the
dependencies. This is not ideal as it will take a long time to build the Docker.

and to build and run the Docker image, you can use the following commands:

```bash title="Build and Run Docker"
#!/bin/bash

# Check if docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker to run this script."
    exit
fi

# Build the Docker image
echo "Building Docker image..."
docker build \
    -t streamlit-app:v1 \
    .

# Run the Docker container
echo "Running Docker container..."
docker run \
    --rm \
    -p 8501:8501 \
    --name streamlit-app \
    streamlit-app:v1
```

Once successfully built and run, you can open your browser and navigate to
`localhost:8501` to see the Streamlit app!
