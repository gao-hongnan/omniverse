# Docker Compose

Check how airbyte defines its docker-compose.yaml.

https://docs.docker.com/compose/ n the context of Docker Compose, the line
db:/var/lib/postgresql/data is mounting a volume to a path inside a Docker
container. Here's what each part means:

db is the name of a Docker volume. This could be a named volume that you defined
in the volumes section of your Docker Compose file, or a volume that Docker
creates automatically with this name.

/var/lib/postgresql/data is the path inside the container where the db volume is
mounted.

In PostgreSQL Docker containers, the /var/lib/postgresql/data directory is where
PostgreSQL stores its data files by default. By mounting a volume to this
directory, you're ensuring that the data stored by PostgreSQL persists even
after the container is removed and recreated, because the data isn't stored in
the container's file system but in the Docker volume.

So, the line db:/var/lib/postgresql/data is telling Docker to mount the db
volume at the location /var/lib/postgresql/data inside the container. This way,
the PostgreSQL data is saved on the db volume and not lost when the container is
restarted or removed.

```yaml
services:
  mlflow:
    image: mlflow-docker-example:1.0.0
    logging: *default-logging
    restart: unless-stopped
    ports:
      - "${MLFLOW_PORT}:${MLFLOW_PORT}"
    environment:
      - MLFLOW_BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI}
      - MLFLOW_ARTIFACT_STORE_URI=${MLFLOW_ARTIFACT_STORE_URI}
    networks:
      - airbyte_internal

  postgres:
    image: postgres:13.2
    logging: *default-logging
    restart: unless-stopped
    ports:
      - "${POSTGRES_PORT}:5432"
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - db_data:/var/lib/postgresql/data
    networks:
      - airbyte_internal
```

This service definition is adhering to the following good practices:

Log management: The logging configuration ensures that logs are properly managed
and don't consume all available disk space.

Restart policy: The restart policy unless-stopped ensures that the service will
always restart unless manually stopped by the user.

Environment Variables: Environment variables are used to make the Docker Compose
file more reusable, and to avoid hardcoding sensitive information such as
usernames and passwords in the file.

Networks: By specifying the networks, you can control the network communication
between services. Here, airbyte_internal network is used.

Volumes: For the postgres service, a named volume db_data is used to persist
database data.

Make sure to replace the placeholders with your actual values. You can add these
values in a .env file in the same directory as your docker-compose.yml file.
Docker Compose automatically picks up the variables defined in this file.

The Docker Compose file you provided makes excellent use of Docker's features
and follows good practices. Here are a few key points:

Environment variables: Environment variables are used extensively to avoid
hardcoding. You can see numerous instances of ${VARIABLE_NAME} in the
docker-compose file. These variables are often set in a .env file that resides
in the same directory as the docker-compose file or set directly in the shell.
These variables can be altered without changing the Docker Compose file itself,
promoting reusability and flexibility.

Docker Volumes: The use of Docker volumes for persisting data is a good
practice. Here, volumes are defined at the end of the docker-compose file
(workspace, data, db) and are attached to specific services. This ensures that
important data is not lost when containers are stopped or removed.

Logging: This docker-compose file uses a logging extension (x-logging) to define
and reuse a common logging configuration across services. The &default-logging
is a YAML anchor that is referenced by \*default-logging in service definitions.

Networks: The file is making use of Docker networking features to isolate and
control communication between the containers. There are two networks defined:
airbyte_internal and airbyte_public.

depends_on: This feature allows you to define the order of service startup and
shutdown.

Service completion: The service init is used as a dependency with condition:
service_completed_successfully for the other services. This is used to make sure
that the initialization is completed successfully before starting other
services.

Service specific environment variables: Each service declares its own set of
environment variables. This provides service-level configuration flexibility and
allows for better security and segregation of service configurations.

Please note that while this Docker Compose file is of a high standard and
utilizes many of Docker's features, it's important to understand the
implications of each setting and adapt them to suit the needs of your project.
This file is written for a specific project (Airbyte) and therefore might
contain some specific configurations that might not be necessary for your
project.

Your original Dockerfile and docker-compose files can be modified to include
environment variables, logging configuration, volumes, and networks in a similar
way. For example, here is how you might modify the PostgreSQL service in your
docker-compose file:
