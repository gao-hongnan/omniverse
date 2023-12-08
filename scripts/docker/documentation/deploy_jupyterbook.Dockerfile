# Start with a Python base image
FROM python:3.9-slim-buster

# Set environment variables
ENV HOME_DIR=/jupyter-book-blog
ENV VENV_DIR=/opt
ENV VENV_NAME=venv

# Create and activate a virtual environment
RUN python -m venv ${VENV_DIR}/${VENV_NAME}
ENV PATH="${VENV_DIR}/${VENV_NAME}/bin:$PATH"

# Set the working directory
WORKDIR ${HOME_DIR}

# Install system dependencies (if any)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    make && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your Jupyter Book content
COPY . ${HOME_DIR}

# Expose the port Jupyter Book uses
EXPOSE 4000

# Default command to build the Jupyter Book
# Can be overridden to serve the book using `jupyter-book build .` followed by `jupyter-book serve .`
CMD ["jupyter-book", "build", "."]


# Copy the entrypoint script
COPY scripts/docker/entrypoint.sh /entrypoint.sh

# Make sure the script is executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
