# Containerization

```{contents}
:local:
```

## Dockerfile

Dockerfile is a text file that contains all the commands a user could call on
the command line to assemble an image.

```Dockerfile
FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y python3 \
    python3-pip

COPY requirements.txt /app
COPY hello_world.py /app

RUN python3 -m pip install -r requirements.txt --no-cache-dir

CMD ["python3", "hello_world.py"]
```

In this short Dockerfile, we already wrote some instructions:

-   `FROM ubuntu:22.04`: This tells Docker to use the base image of Ubuntu 22.04
    as the starting point for building our image.
-   `WORKDIR /app`: This sets the working directory to `/app`. All subsequent
    instructions will be executed in this directory.
-   `RUN apt-get update -y && apt-get install -y python3 python3-pip`: This runs
    two commands in a single layer. It updates the package list and installs
    Python 3 and pip.
-   `COPY requirements.txt /app`: This copies the `requirements.txt` file from
    the host machine to the container's working directory.

## What is an image?

A Docker image is a lightweight, standalone, executable package that includes
everything needed to run a piece of software. Images are created from a
Dockerfile, a text file that contains a list of commands to assemble an image.
Images are the blueprint for containers, and they become containers when they
run on Docker Engine.

Each instruction in a Dockerfile creates a **read-only** layer in the image.
When you change the Dockerfile and rebuild the image, only those layers that
have been changed are rebuilt. This is part of what makes Docker images so
lightweight, small, and fast when compared to other virtualization technologies.

## What are layers?

Each instruction in a Dockerfile creates a new layer in the Docker image. Layers
are stacked on top of each other to create the final Docker image. Each layer is
only a set of differences from the layer below it. These layers are read-only.

Let's consider our Dockerfile earlier.

-   `FROM ubuntu:22.04`: This is the base layer of your image, it's a snapshot
    of the filesystem of Ubuntu 22.04.
-   `WORKDIR /app`: This changes the working directory to `/app`. It effectively
    adds a new layer to the image, which is the filesystem of the previous layer
    plus the change of the working directory.
-   The `RUN` command that updates and installs Python3 adds another layer. This
    layer includes all the filesystem of the previous layers plus the changes
    made by the update and install operations.
-   The `COPY` commands to copy `requirements.txt` and `hello_world.py` to
    `/app` create new layers with the added files.
-   The `RUN` command that installs the requirements using `pip` adds another
    layer. This layer includes the filesystem of the previous layers plus the
    changes made by the install operation.
-   Finally, the `CMD` command doesn't change the filesystem, so it doesn't
    create a new layer.

> You need to think of a container as your computer, you can run your training
> pipeline because of? Because of your python scripts, because of your python
> executable, because of libraries and operating system. All of these "need to
> be moved to the container somehow". So you can think of filesystem as a
> "container" for all of these things. And each layer is a "snapshot" of the
> filesystem.

## Why are layers read-only?

Think of an image consisting multiple layers stacked from bottom to top, where
bottom is the first line of instruction in the `Dockerfile`, `FROM ubuntu:22.04`
in our case. Now actually this layer in itself has many layers since it is built
on top of a base image.

Now imagine if the layers are not read-only, and you make a change in the
running container.

-   Consistency: Say you go into the container and change the `requirements.txt`
    file to empty. Now if container A and B are built on top of the same image,
    and you run container A and change the `requirements.txt` file to empty,
    then container B will also have the `requirements.txt` file empty. This is
    because the `requirements.txt` file is not read-only, and the change you
    made in container A is reflected in container B as well.

-   Loss of Reproducibility: If a container could modify its base image layers,
    then the same image could behave differently on different hosts or at
    different times, depending on what changes had been made by containers.

An immediate question is, when using docker, I do encounter situations where the
container seemingly alters the image. For example, when I run a container and
the code output a lot of files, I can see those files in the image. How is this
possible? We answer it in the next section.

## What is container layer?

A container layer, also known as the writable layer, is a layer added on top of
all the read-only image layers when a Docker container is started from an image.
This layer is unique to the container and is the only part of the filesystem
that the container can write to. It's what allows a running container to appear
as a normal, fully writable Linux system to the processes running inside it.

"Changes in a running container" refers to any modifications to the filesystem
made by the processes inside the container during its lifetime. This can include
a wide range of operations, such as:

-   Creating, modifying, or deleting files and directories. For example, an
    application might write logs to a file, or a database might write data to
    its storage files.
-   Installing additional software packages, libraries or dependencies.
-   Changing configuration settings or environment variables.

All of these changes are written to the container's writable layer. However, by
default, these changes are not persistent -- if the container is stopped and
removed, its writable layer is also removed, and the changes are lost.

If you want to keep these changes, you can do one of the following:

-   Commit the changes to a new image: Docker allows you to commit the state of
    a container's filesystem as a new image. This essentially creates a new
    read-only layer that encapsulates all the changes made in the container
    layer. You can then start new containers from this image, which will include
    the changes.

-   Use Docker volumes or bind mounts: These are mechanisms for persisting data
    generated by and used by Docker containers. Volumes are stored in a part of
    the host filesystem which is managed by Docker
    (`/var/lib/docker/volumes/...`), and bind mounts can be anywhere on the host
    filesystem. These provide a way to share data between the host and the
    container, or between multiple containers, and to keep this data even after
    the container is removed.

## Rebuilding layers and cache

When you build an image, Docker will try to use cached layers as much as
possible. This means that if you change a line in your Dockerfile, only the
layers after that line will be rebuilt. This is why it's important to order your
Dockerfile instructions from least to most likely to change.

So here new layers are created or replaced the old ones when there is a change
in the Dockerfile. The distinction between here is that you are rebuilding the
image where previously you were just running the container and hence the layers
are read-only.

---

When a Docker image is built from a Dockerfile, Docker performs each instruction
in the Dockerfile one-by-one, committing the result of each instruction to a new
layer.

When Docker rebuilds an image, it uses a caching mechanism to speed up the
building process. It examines each instruction in the Dockerfile and looks for
an existing layer in its cache that resulted from the same instruction. If it
finds one, it just reuses that layer instead of creating a new one.

However, if an instruction changes, Docker can no longer use the cached layer
for that instruction or any subsequent ones, because the cached layer doesn't
reflect the result of the new instruction. Instead, it executes the new
instruction and creates a new layer for it, and for any instructions that follow
it, because their results could depend on the result of the changed instruction.

So, while each layer in an image is read-only and doesn't change once it's
created, Docker can create new layers when the Dockerfile instructions change.
These new layers replace the corresponding layers in the image, which gives the
appearance that the layers have changed.

For example, let's say you have a Dockerfile with three instructions: A, B, and
C. The first time you build an image from this Dockerfile, Docker creates three
layers: Layer A from instruction A, Layer B from instruction B, and Layer C from
instruction C.

Now, you change instruction B to B'. When you rebuild the image, Docker reuses
Layer A from its cache, because instruction A hasn't changed. But it can't reuse
Layer B or Layer C, because their corresponding instructions have changed or
could be affected by the change. So, it executes instructions B' and C and
creates new layers for them, Layer B' and Layer C'.

The resulting image still has three layers, but Layers B and C have been
replaced with new layers that reflect the updated instructions. Even though the
layers are read-only and don't change themselves, Docker can create new layers
to reflect changes in the Dockerfile instructions.

## What is a container?

-   A container is an isolated environment that holds the components needed to
    run a piece of software, including the code, runtime, system tools,
    libraries, and settings. Containers are isolated from each other and bundle
    their own software, libraries, and configuration files; they can communicate
    with each other through well-defined channels.

In the context of Docker, a container is created from an image. So a container
is a running image.

## What is a filesystem?

The term "filesystem" refers to the methods and data structures that an
operating system uses to manage and keep track of files on a disk or partition;
that is, the way the files are organized on the disk. The term is also used to
refer to a partition or disk that is used to store the files or the type of the
filesystem.

Let's break it down:

**File:** The basic unit of storage. Files can store text, data, image, video,
program instructions, etc. Each file is associated with a set of attributes such
as its name, location, size, creation time, modification time, and permissions.

**Directory (or Folder):** A special type of file that contains a list of other
files and directories.

**Filesystem:** The overall structure in which files are named, stored, and
organized. A filesystem manages the storage and retrieval of data from the
storage medium (like a hard drive). It organizes the files and directories into
a hierarchical structure, keeps track of where data for a file is stored, and
manages metadata associated with the files, like permissions, timestamps, and
ownership information.

For example, in a Linux-based filesystem, everything starts from a root
directory (designated as "/"), and then expands into sub-directories and files.
So, a file might have a path like `/home/user/document.txt`, where `/home` is a
subdirectory of `/`, and `user` is a subdirectory of `/home`, and `document.txt`
is a file in the `user` directory.

When we talk about the "filesystem" in a Docker image, we're referring to all
the files and directories that the image contains, which could include the
operating system files, application code, dependencies, configuration files,
etc. These files are structured in a hierarchical manner, similar to the way
files are organized on a regular computer system.

## What is a [kernel](<https://en.wikipedia.org/wiki/Kernel_(operating_system)>)?

The kernel is a core component of an operating system. It works as a bridge
between applications and the actual data processing done at the hardware level.
The kernel's responsibilities include managing the system's resources (the
communication between hardware and software components) and providing essential
services for the execution of applications and processes.

Here's an intuitive way to understand the kernel:

Imagine you're in a busy kitchen. You (the user) are trying to cook several
recipes (run applications) at once. You have a variety of kitchen appliances and
tools (hardware) like an oven, a blender, a refrigerator, etc.

Now, you could try to manage all these tasks yourself - keep track of cooking
times, adjust temperatures, mix ingredients, clean tools, and so on. But it's a
lot to handle at once, and if you're not careful, you might burn something or
make a mistake.

So, you bring in a chef (the kernel). The chef is an expert at managing the
kitchen. You give the chef your recipes, and the chef takes care of everything
else. The chef decides when to use the oven, how to mix the ingredients, when to
clean up, etc. All you need to do is tell the chef what you want, and the chef
makes it happen.

In this analogy:

-   The chef (kernel) is the part of the system that handles all the complex
    management tasks.
-   The recipes (applications) are what you (the user) want to accomplish.
-   The kitchen appliances (hardware) are the resources the chef uses to cook
    the recipes.

In essence, the kernel is the "master chef" of your computer system. It handles
all the low-level tasks, like managing hardware resources and running
applications, so you don't have to worry about them.

## How is Docker and Kernel related? Important but difficult

Docker leverages the Linux kernel features to provide its functionalities. Some
of the key features are:

1. **Namespaces**: Docker uses namespaces to provide isolated workspaces called
   containers. When you run a container, Docker creates a set of namespaces for
   that container which provides a layer of isolation: each container runs
   within its own namespace and does not have access outside it. For example,
   pid (processes), net (network), mnt (mount points), and ipc (inter-process
   communication) are some of the namespaces Docker uses.

    See my earlier definition of namespace.

2. **Control Groups (cgroups)**: Docker uses cgroups to limit and isolate the
   resource usage (CPU, memory, I/O, network) of containers. This way, Docker
   ensures no single container takes up all the system resources.

3. **Union File Systems (UnionFS)**: Docker uses UnionFS to create layers, which
   makes Docker lightweight. UnionFS allows files and directories of separate
   file systems (known as branches), to be overlaid, forming a single coherent
   file system.

4. **Capabilities**: Docker uses capabilities to provide a finer level of access
   control to the Docker host's resources. By default, Docker drops all
   capabilities except those needed by a container to function.

5. **Networking**: Docker uses virtual network interfaces and routing rules to
   isolate network traffic between containers.

All these features are provided by the Linux kernel, and Docker leverages them
to provide a powerful platform for deploying and running applications inside
containers. In simple words, Docker can be seen as a high-level API on top of
the Linux kernel features.

### What is a Namespace?

Namespaces are a feature of the Linux kernel that provide isolation for certain
system resources, so that processes within each namespace have their own
isolated instance of these resources. Namespaces form the foundation of many
types of containerization, as they allow each container to have its own isolated
set of process IDs, network stacks, file systems, and other resources.

Types of namespaces include: PID (Process IDs), USER (User and group IDs), UTS
(Hostname and domain name), NET (Network devices, stacks, and ports), MNT (Mount
points and filesystems), IPC (Inter-process communication resources), and CGROUP
(Control group resources).

**In a more rigorous way:** Namespaces are a feature of the Linux kernel that
partitions kernel resources such that one set of processes sees one set of
resources while another set of processes sees a different set of resources. The
feature works by having the same namespace for a set of resources and processes,
but those namespaces are distinct from other namespaces that are associated with
other sets of resources and processes.

**In layman's terms:** Imagine you are in a large office building where many
companies have their offices. Each company can only see and interact with their
own office space - they can't see into the offices of other companies in the
building. In this analogy, the office building is the Linux system, each company
is a different process or group of processes, and each office is a namespace.
Just like how the walls of each office isolate each company from one another,
namespaces in Linux isolate different groups of processes from one another. They
all share the same "building" (the Linux system), but their view of that system
is isolated to their own "office" (the namespace).

So:

-   Office building = Linux system
-   Company = Process
-   Office = Namespace

For instance, if two processes are in different PID namespaces, each could have
a process with PID 1, without conflict. Each process would only see their own
processes within the same namespace. If they tried to look for a process from a
different namespace, it would be as if that process did not exist. It's a bit
like being in parallel universes - each process lives in its own universe,
unaware of processes in different universes.

### What is a Control Group?

Rigor: Control groups (cgroups) is a Linux kernel feature that allows Docker to
share available hardware resources to containers and, if required, set up limits
and constraints. For example, limiting the memory available to a specific
container.

Layman: Imagine a buffet dinner where everyone has access to all the dishes. But
to make sure that everyone gets a fair share, and no one ends up eating all the
dessert, there are rules. These rules specify how much each person can take from
each dish. That's essentially what control groups do. They make sure that each
container (person) gets their fair share of the system resources (food), and no
single container can hog all resources.

### What is a Union File System?

Rigor: UnionFS is a filesystem service for Linux, FreeBSD and NetBSD which
implements a union mount for other file systems. It allows files and directories
of separate file systems, known as branches, to be overlaid, forming a single
coherent file system. This allows Docker to build up containers in layers and
share common files, reducing the footprint of the system.

Layman: Imagine a stack of transparent papers, each containing a part of a
complete picture. Individually, these papers are not much use, but when you
stack them correctly, you see the full image. That's what UnionFS does. It
combines different file systems (transparent papers), each possibly having
different parts of the system files (parts of the image), into one unified file
system (the complete picture), making it seem as if all the files exist on a
single layer.

### What is a Capability?

Rigor: Capabilities are a kernel feature that breaks down the privileges of the
root user into a set of distinct privileges, or "capabilities," that can be
independently enabled or disabled. Docker uses capabilities to provide a finer
level of access control to system resources and operations.

Layman: Suppose you have a super key to a large building that can open every
single door. But for security reasons, you don't want to give a full super key
to everyone. Instead, you break the super key into smaller keys, each of which
can only open certain doors. This is what capabilities do, they break down the
superpowers of the root user into smaller, more specific privileges.

### What is Networking?

Rigor: Docker uses network namespaces and a virtual network interface to provide
each container with its own network stack. This allows each container to have
its own view of the network, the network devices, IP addresses, and routing
tables, thereby providing isolation and control over network traffic between
containers.

Layman: Suppose you're in an apartment complex where each apartment has its own
private Wi-Fi network. Even though the same internet service provider may serve
the entire complex, each apartment's Wi-Fi network is isolated from the others.
You can't see or connect to your neighbor's devices and vice versa. Docker's
networking works similarly by providing each container with its own isolated
network.

## Analogy

### Analogy 1

This analogy can be useful in understanding the relationship between Docker
images and Docker containers, but there are some important distinctions to
consider.

1. **Docker image as a class**: This comparison works because a Docker image is
   a template or blueprint for creating Docker containers, much like a class is
   a blueprint for creating objects in object-oriented programming. A Docker
   image defines the filesystem and configuration of the containers created from
   it, similar to how a class defines the attributes and methods of its
   instances.

2. **Docker Layers as Class Methods**: Docker layers could be likened to the
   methods in a class definition. Each Docker layer represents a set of changes
   or instructions that are applied to the filesystem (akin to a method
   performing an operation), and they are executed in sequence to construct the
   Docker image. This is similar to how methods in a class define different
   operations that can be performed on or by instances of the class. However,
   it's important to remember that Docker layers are immutable once created,
   unlike methods which can be executed with different contexts and inputs.

3. **Docker container as an instance of a class**: This is also a good analogy.
   A Docker container is a runnable instance of a Docker image. It has its own
   state and can have different data, much like an object is an instance of a
   class, with its own state and potentially different attribute values.

4. **Container Layer (Writable Layer) as Instance Variables**: When a container
   is run from an image, it adds a new, writable layer on top of the image
   layers. This could be compared to instance variables in an object. The
   writable layer is unique to each container (like instance variables are
   unique to each object), and it's where any changes (like creating, deleting,
   or modifying files) are made during runtime. This is similar to how instance
   variables store the state that can change during the lifetime of an object.

### So multiple containers can share the same image but do different things. Why?

When you create multiple instances of a class, the objects are structurally
identical because they're created from the same blueprint. Similarly, when you
create multiple Docker containers from the same Docker image, the initial state
of each container is identical to the others because they're created from the
same image.

However, after a container (or an object instance) is created, it has its own
life and can change independently of other containers (or other object
instances).

For example, suppose you have a class in Python representing a bank account:

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            print("Insufficient funds!")
```

You might create multiple instances of the `BankAccount` class:

```python
account1 = BankAccount("Alice", 100)
account2 = BankAccount("Bob", 200)
```

At the moment of creation, `account1` and `account2` are separate instances with
the same structure, but different states (different owners and balances). As you
call methods like `deposit` and `withdraw` on these instances, their state will
change independently of each other.

Similarly, when you create a Docker container from an image, it starts in the
state defined by the image, but as it runs, it can change independently. For
example, if the container runs a web server, each container might handle
different requests and therefore have different logs. If the container runs a
database, each container will have its own unique data.

Therefore, while Docker containers created from the same image start in the same
state, their states can diverge as they run, much like instances of a class can
have different states.

### So multiple containers can share the same image but do different things. How?

A Dockerfile indeed provides a set of instructions to create a Docker image,
which is essentially a snapshot of a filesystem and certain configuration
values. When you start a container from an image, it's like starting from that
snapshot, so all containers started from the same image will start in the same
state.

However, what happens next can vary between containers. This is because a
container isn't just a static filesystem; it's a running environment where you
can start processes. The processes that run in a container can interact with
each other, the filesystem, the network, and in some cases, external systems
like databases or other web services.

For example, let's consider a Dockerfile that sets up a simple web server:

```Dockerfile
FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
```

Suppose `app.py` is a simple web application that logs every request it receives
to a file. Now, imagine you build an image from this Dockerfile and start two
containers from that image. Each container starts in the same state, with the
same files and the same web server.

But as soon as the containers start receiving requests, their state will begin
to diverge. One container might receive a GET request to the `/home` endpoint,
while the other might receive a POST request to the `/login` endpoint. The log
file in each container will reflect the requests that that specific container
handled, and so the filesystem in each container (specifically, the log file)
will be different.

Here's another example. Suppose you have a Dockerfile that sets up a simple
script that writes the current time to a file:

```Dockerfile
FROM ubuntu:20.04

WORKDIR /app

COPY script.sh /app

CMD ["bash", "script.sh"]
```

And `script.sh` is as simple as:

```bash
#!/bin/bash

date > timestamp.txt
```

If you build an image from this Dockerfile and start two containers a few
minutes apart, the `timestamp.txt` file in each container will contain a
different time, because the `date > timestamp.txt` command was run at different
times in each container.

**In both of these examples, the Dockerfile and the resulting image are the same
for both containers, but the state of the containers can diverge due to the
actions of the running processes within each container. This illustrates how
Docker containers, like instances of a class, can have different states even
when they're created from the same image (or class definition).**

### Analogy 2

Here's a refined analogy:

-   **Docker Image as a Blueprint**: Think of a Docker image like a blueprint
    for a house. It includes all the details on how to construct the house, just
    like a Docker image includes all the filesystem details and configurations
    necessary to create a container.

-   **Docker Container as a House**: A Docker container is like a house built
    from the blueprint. It's a running instance of an image. You can live in it
    and change the interior as you wish, which is analogous to running processes
    in the container and making changes in its writable layer.

-   **Docker Layers as Construction Steps**: The layers of a Docker image could
    be seen as the different stages of the house construction process. One layer
    might be the foundation, the next could be the framing, then the plumbing,
    electrical work, etc. Each layer adds something to the overall structure,
    just as each Docker layer adds something to the image (like a software
    package or a file). And each subsequent layer depends on the layers beneath
    it, just as each step in building a house depends on the previous steps.

-   **Container Layer (Writable Layer) as Interior Design**: The container layer
    can be seen as the interior design of the house. It's the space where you
    can make changes without affecting the structure of the house. In a
    container, this layer is where changes (like creating, deleting, or
    modifying files) are made during runtime.

This analogy captures the concepts a bit more accurately while maintaining the
basic comparison to object-oriented programming.

## Docker Networking

### Can you explain Docker Networking? What types of network drivers does Docker support?

TODO: Add more details

In the context of Docker, understanding the `bridge`, `host`, and `overlay`
networks are arguably the most important for a solid foundational understanding.
This is because these network types cover a broad range of use cases that you'll
commonly encounter in real-world scenarios.

-   The `bridge` network is the default network type for containers, and
    understanding it will help you understand how containers communicate with
    each other and with the host on the same machine.

-   The `host` network is useful when you want to avoid the network isolation
    provided by Docker for performance reasons or when dealing with specific
    networking requirements.

-   The `overlay` network is crucial when working with Docker in a distributed,
    multi-host environment, such as a Docker Swarm.

Let's see a basic example for each:

1. **Bridge Network**

    The bridge network is the default network type when you create a new
    container. To demonstrate its use, we can start two containers on the same
    bridge network and see how they can communicate with each other.

    ```bash
    docker run -d --name container1 --network bridge busybox sleep 1d
    docker run -d --name container2 --network bridge busybox sleep 1d
    docker exec container1 ping -c 3 container2
    ```

    In this example, `container1` is able to ping `container2` using its name,
    showing that they're able to communicate over the bridge network.

2. **Host Network**

    With host networking, a container shares the host's network stack, and does
    not get its own IP address allocated.

    ```bash
    docker run --network=host -d nginx
    ```

    In this case, the nginx server is accessible on `localhost` (or the host's
    IP) because it's using the host's network.

### Follow up: why do you need network?

Let's illustrate Docker networking using a simple web application example.

Let's say you're running a web application that consists of three components:

-   A front-end web server (let's say it's running Nginx)
-   An API server (a Python Flask app, for example)
-   A database (like PostgreSQL)

Each of these components runs in its own Docker container. However, these
containers need to communicate with each other for the application to work. The
front-end needs to call the API server, and the API server needs to read and
write data from and to the database.

Here's how Docker networking enables this:

1. **Inter-Container Communication (BRIDGE)**: You create a bridge network
   (let's call it `my-network`). When you start the containers, you connect them
   to this network. On this network, containers can communicate with each other
   using their container names as hostnames.

    ```bash
    # Create the network
    docker network create my-network

    # Start the containers on my-network
    docker run --network my-network --name db -d postgres
    docker run --network my-network --name api -d my-flask-app
    docker run --network my-network --name web -d nginx
    ```

    Now, in your Flask app, you can use `db` as the hostname to connect to the
    PostgreSQL database. And in your Nginx configuration, you can use `api` as
    the hostname to proxy requests to the Flask app.

2. **Communication with the Outside World (HOST)**: For the outside world to
   access your application, you need to map a port on your host machine to the
   Nginx container.

    ```bash
    # Start the Nginx container with port mapping
    docker run --network my-network --name web -d -p 80:80 nginx
    ```

    Now, requests to `http://your-host-ip` in a web browser will be forwarded to
    the Nginx container, which can then forward the requests to the Flask app
    via the bridge network.

This example illustrates how Docker networking enables both inter-container
communication and communication between containers and the outside world. It's a
simple scenario, but real-world applications will often follow a similar
pattern, albeit with more containers and potentially more complex networking
setups.

### Follow up: similar to docker-compose

```yaml
version: "3"

services:
    db:
        image: postgres
        networks:
            - my-network

    api:
        image: my-flask-app
        networks:
            - my-network
        depends_on:
            - db

    web:
        image: nginx
        ports:
            - "80:80"
        networks:
            - my-network
        depends_on:
            - api

networks:
    my-network:
        driver: bridge
```

### Follow up: when do you need specify --network=host?

The `--network=host` option in Docker is used to set the networking mode for the
container to the host mode. In this mode, the container shares the network stack
with the host system and does not get its own isolated network environment. This
means that the container has access to the host's network interfaces and can
listen to or connect to ports directly on the host's IP address.

Here are a few situations when you might want to use `--network=host`:

1. **Performance**: In some cases, network performance is critical and the
   slight overhead of the bridge network can be a problem. For instance, high
   performance computing or real-time applications may require this.

2. **Listening to network traffic**: If the application in the container needs
   to monitor or capture network traffic from the host system, it needs access
   to the host's network stack. This is common in network analysis or security
   tools.

3. **Distributed Systems Testing**: If you're testing a distributed system
   locally, you might run each node in a Docker container and have them all
   communicate over the localhost network.

4. **Simplicity**: If your container needs to use the same ports as the host or
   if there are many ports that need to be exposed, using `--network=host` can
   be simpler than mapping each port individually.

5. **Access to host network services**: If the container needs to connect to
   services on the host system that aren't exposed over the network, host
   networking can be used.

However, keep in mind that `--network=host` essentially disables the network
isolation between the container and the host system. This goes against the
principle of isolating the application environment that Docker is built around,
and can potentially pose a security risk. Hence, its usage should be carefully
evaluated.

Furthermore, `--network=host` mode behaves differently on Windows and MacOS
Docker hosts due to how networking is implemented for Docker on those operating
systems. Therefore, its use may lead to inconsistencies or unexpected behaviors
if your application is intended to run on multiple platforms.

## Docker Storage

See my iris example!

<https://docs.docker.com/storage/>

### What is a docker volume?

Docker volumes are a mechanism for persisting data generated by and used by
Docker containers. They are essentially directories (or files) that live on the
Docker host filesystem, and are accessible (i.e., can be read from or written
to) by containers.

Volumes are designed to solve the problem of persisting data between container
runs. Remember that containers are ephemeral and stateless; when a container is
destroyed, all its filesystem changes are lost. If a container's job involves
modifying data (like a database, for instance), it's essential to have a way to
persist that data. This is where Docker volumes come in.

### How can Docker volumes be shared among multiple containers?

Docker volumes can also be shared among multiple containers. This is useful for
cases where more than one container needs to access or modify the same set of
data.

To use a volume, you first create it:

```bash
docker volume create my-volume
```

Then you can attach it to containers at runtime:

```bash
docker run -d --name container1 -v my-volume:/app/data my-image
docker run -d --name container2 -v my-volume:/app/data my-image
```

Here, `-v my-volume:/app/data` means "mount the volume `my-volume` at the path
`/app/data` inside the container". Both `container1` and `container2` have
access to the same data, and any changes one container makes will be visible to
the other.

Note that this kind of volume can be used by multiple containers simultaneously.
This can be useful for scenarios such as having a worker container process some
data and then another container take over from where the worker left off. But
it's also something to be cautious of; if multiple containers are trying to
write to the same volume simultaneously, conflicts can occur.

### What is a bind mount?

### What is the difference between a Docker volume and a bind mount?

## How would you monitor the performance of Docker containers?

Monitoring the performance of Docker containers is a critical aspect of managing
a Docker-based environment. Docker includes several tools for monitoring the
performance of containers, and many third-party tools are also available. Here's
a basic overview:

1. **Docker stats command**: Docker's built-in tool for monitoring containers is
   the `docker stats` command. It displays a live data stream for running
   containers, including metrics like CPU usage, memory usage, network I/O, and
   disk I/O. The `docker stats` command is a good place to start for basic
   monitoring needs.

    ```bash
    docker stats
    ```

2. **Docker logs command**: This is another Docker built-in command that can
   provide insight into a container's behavior. This command shows you the logs
   of a container, which is very useful for debugging purposes.

    ```bash
    docker logs [container_id]
    ```

3. **Docker Remote API**: Docker provides a Remote API that you can use to
   gather even more in-depth information about your containers. This can be
   useful if you need to integrate Docker with other monitoring systems or
   custom-built tools.

4. **Third-party monitoring tools**: There are many third-party tools available
   for more advanced monitoring. Some popular ones include:

    - **Prometheus**: An open-source monitoring solution that integrates well
      with Docker and Kubernetes. It allows you to query and visualize metrics
      using a robust query language.
    - **Grafana**: An open-source platform for metrics visualization. It's often
      used alongside Prometheus to create dashboards for monitoring container
      performance.
    - **cAdvisor**: An open-source tool for monitoring container metrics,
      developed by Google.
    - **Datadog**: A commercial platform offering powerful features for logging,
      monitoring, and visualizing metrics.

5. **Docker Health Checks**: You can also specify health checks in Docker to
   monitor the health status of services running inside the containers. Docker
   will execute the health checks at regular intervals, and report the health
   status of the container.

Remember that which tool or approach you choose will depend on your specific
needs, the complexity of your environment, and your operational resources. It's
often beneficial to have a combination of several of these tools to ensure you
have full visibility into your Docker containers' performance.

## Can you discuss some best practices for writing a Dockerfile?

Writing efficient, clear, and secure Dockerfiles is crucial to building
effective Docker images. Here are some best practices for writing Dockerfiles:

1. **Minimize the number of layers**: Each instruction in a Dockerfile creates a
   new layer in the Docker image. Therefore, to reduce the number of layers and
   the overall size of the image, it's a good practice to chain commands
   together in a single `RUN` instruction using `&&`. However, remember that
   readability and maintainability is also important.

2. **Use a .dockerignore file**: Just like .gitignore, .dockerignore file can
   exclude files not relevant to the build (e.g., logs, caches, local
   environment variables files) from your Docker build context. This helps to
   improve build performance and reduce the image size.

3. **Use multi-stage builds**: Multi-stage builds allow you to effectively
   separate the build-time dependencies from the runtime dependencies, resulting
   in smaller Docker images. They can be especially useful when building
   applications in compiled languages.

4. **Don’t run as root**: If a user can break out of your application, they'll
   be within your container, as the root user, which can lead to significant
   security issues. Hence, it's a good practice to run processes as a non-root
   user in the container, even if the container is isolated.

5. **Use COPY instead of ADD**: Unless you specifically need the additional
   capabilities of ADD (like remote URL support and automatic tar extraction),
   it's recommended to use COPY as it's more transparent and predictable.

6. **Use specific tags in the FROM command**: Using the `latest` tag or no tag
   in the FROM command makes it unclear what version of the image is being used.
   This can lead to inconsistent behavior. It's better to use specific version
   tags.

7. **Set a working directory**: It's a good practice to set a working directory
   (using the `WORKDIR` instruction) in which you'll operate. This way, you
   won’t have to use absolute paths.

8. **Clean up after installation commands**: If you’re installing packages in
   your Dockerfile, make sure to clean up the cache after the installation step
   to keep your layers (and overall image size) as small as possible.

9. **Leverage build cache**: Docker caches the result of each Dockerfile
   instruction to speed up subsequent builds. But Docker invalidates the cache
   for an instruction if the instructions before it have changed. Hence, it's
   better to order the Dockerfile instructions with the least frequently
   changing ones at the top.

10. **Add Healthchecks**: Including healthchecks in your Dockerfile can make
    your containers more robust and self-healing by allowing Docker to know the
    status of the applications running within your containers.

These are some general best practices, but the specifics may depend on the
particular needs of your project or organization.

## Docker Compose vs Docker Swarm

Docker Swarm and Docker Compose are both orchestration tools provided by Docker,
but they serve different purposes and are used in different contexts.

**Docker Compose**:

Docker Compose is a tool for defining and running multi-container Docker
applications. With Compose, you define your application's services, networks,
and volumes in a single YAML file, then start all the services with a single
command (`docker-compose up`).

Compose is great for development and testing, and for continuous integration
workflows. It simplifies the process of running complex applications consisting
of multiple interconnected containers on a single machine. However, it doesn't
provide any facilities for scaling or managing your application across multiple
machines.

**Docker Swarm**:

Docker Swarm, on the other hand, is a clustering and scheduling tool for Docker.
It allows you to create and manage a swarm – a group of machines that are
running Docker and joined into a cluster.

Once you've created a swarm, you can define services – essentially distributed
applications consisting of multiple containers – that Docker will automatically
distribute across the swarm, handling replication, service discovery, load
balancing, and scaling for you. Swarm also provides a degree of fault tolerance,
automatically rescheduling containers if a node fails.

Swarm is intended for managing applications across multiple machines in a
production environment. It's more complex to set up and use than Compose, but
provides powerful features for system administrators and is capable of managing
large deployments.

In summary, Docker Compose is primarily aimed at defining and running
multi-container applications on a single machine, and is particularly
well-suited to development environments. Docker Swarm is intended for deploying,
scaling, and managing multi-container applications across a cluster of machines,
and is better suited to production environments.

It's worth noting that with the introduction of the Docker stack deployment
feature, you can now use a Docker Compose file to deploy services in Swarm mode,
allowing you to use the same file to define your application in both development
(with Docker Compose) and production (with Docker Swarm).

## When is the -p used in docker run?

The `-p` option in the `docker run` command is part of Docker's networking
features. By default, Docker uses a bridge network, which is a private internal
network created by Docker on the host. Containers can communicate with each
other if they are connected to the same bridge network, and they can also
communicate with the host.

When you use the `-p` option like `-p 8501:8501` (HOST:CONTAINER), you're asking
Docker to forward traffic incoming on the host's port 8501 to the container's
port 8501. This allows external applications, such as your web browser, to
communicate with the application running inside the Docker container.

This uses the Docker's bridge network under the hood. The host's port 8501 is
accessible to outside world, and the communication between the host's port 8501
and container's port 8501 happens over the bridge network.

There's also the host network, which is different. If you start a container and
specify it to use the host network by using the `--network=host` flag in the
`docker run` command, the container shares the network stack with the host and
does not get its isolated network stack. Therefore, if a service in the
container is listening on port 8501, it's directly accessible on the host's port
8501 without needing any port mapping. However, this mode is generally used less
frequently due to the reduced network isolation.

If you do not include the `-p 8501:8501` option, port 8501 in the Docker
container will not be accessible from the host machine. This means that if your
Streamlit app is running on port 8501 inside the container (which it does by
default), you will not be able to access it from your host machine's web
browser.

In other words, the Streamlit app will still be running inside the Docker
container, but you won't be able to interact with it through your host machine's
network.

## References and Further Readingss

-   [Docker Image vs Container](https://circleci.com/blog/docker-image-vs-container/)
