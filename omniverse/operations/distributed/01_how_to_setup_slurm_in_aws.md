# How to Setup SLURM and ParallelCluster in AWS

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

This guide will help you set up your AWS environment and install AWS CLI and AWS
ParallelCluster

## Setting Up Identity and Access Management (IAM) Role

Before deploying AWS ParallelCluster, it's essential to configure an IAM role
with appropriate permissions. This role ensures that AWS ParallelCluster has the
necessary permissions to manage AWS resources on your behalf, such as EC2
instances, networking components, and storage systems.

The rough steps are as follows:

-   Navigate to the [IAM console](https://console.aws.amazon.com/iam/) in AWS.
-   Create a new IAM user or use an existing one.
-   Ensure the user has an IAM role with `AdministratorAccess` or attach the
    `AdministratorAccess` policy directly. While convenient, using
    `AdministratorAccess` can pose security risks if not managed carefully. It's
    recommended to scope down the permissions to least privilege principles if
    you are working in a team or production environment. For educational
    purposes, it's fine to use `AdministratorAccess`.
-   Generate an **AWS Access Key ID** and **AWS Secret Access Key** for
    command-line access - you can find them in the user's security credentials
    tab in the IAM console.

## Configure AWS CLI

First we install the python package `awscli` using `pip`.

```bash
❯ pip install --upgrade awscli
```

Then we configure the AWS CLI:

```bash
❯ aws configure
AWS Access Key ID [None]: <AWS_ACCESS_KEY>
AWS Secret Access Key [None]: <AWS_SECRET_KEY>
Default region name [None]: ap-southeast-1
Default output format [None]: json
```

Replace `<AWS_ACCESS_KEY>` and `<AWS_SECRET_KEY>` with your actual AWS
credentials. And since we want a region close to home we set `ap-southeast-1` as
the default region - it is also recommended to set up the region close to you to
reduce latency.

## Create EC2 Key Pair

To set up the SLURM cluster, we need to create an EC2 key pair. This key pair
can be viewed as a secure SSH access to the cluster's head and compute nodes
where authorized users can log in to train models, run experiments, and manage
the cluster.

```bash
❯  aws ec2 create-key-pair --key-name <YourKeyName> --query 'KeyMaterial' --output text > ~/.ssh/<YourKeyName>.pem
```

We also set the permissions of the key pair file to `600` to ensure that only
the owner can read and write to the file.

```bash
❯ chmod 600 ~/.ssh/<YourKeyName>.pem
```

## Configure AWS ParallelCluster

First, we install the `aws-parallelcluster` package using `pip` so that we can
use the `pcluster` command.

```bash
❯ pip install -U aws-parallelcluster
```

```bash
❯ pcluster configure --config config.yaml
```

### 1. Review the Configuration File

Before proceeding, it’s a good idea to double-check your `config.yaml` file.
Make sure that all settings such as instance types, network configurations,
placement groups, and other parameters accurately meet your needs. Adjust any
settings if necessary. For example, I expected the `HeadNode` to be `t2.small`
and indeed it was.

```yaml
HeadNode:
    InstanceType: t2.small
```

### 2. Create the Cluster

Run the following command to create the cluster. Replace `cluster-name` with
your desired name for the cluster:

```bash
pcluster create-cluster --cluster-configuration config.yaml --cluster-name <cluster-name> --region ap-southeast-1
```

This command specifies the cluster configuration file (`config.yaml`), the name
of the cluster, and the AWS region where you want the cluster to be deployed.

### 3. Monitor the Cluster Creation

The cluster creation process may take some time. You can monitor the progress by
running the following command:

```bash
pcluster describe-cluster --cluster-name <cluster-name> --region ap-southeast-1
```

## Logging into the Head Node

```bash
pcluster ssh --cluster-name <cluster-name> -i <path-to-your-key.pem> --region ap-southeast-1
```

### Compute Node

Due to quota limit, our compute node is `g4dn.xlarge` and has only 1 GPU per
node. If we want 2 GPUs per node, we can use `g4dn.2xlarge` but we need to
request a limit increase.

### SLURM Status

And we can check status of the slurm cluster:

```bash
❯ sinfo

PARTITION          AVAIL  TIMELIMIT  NODES  STATE NODELIST
distributed-queue*    up   infinite      2   idle distributed-queue-st-g4dnxlarge-[1-2]
```

And we can see the nodes are up and running. We can use `srun` to see if the
nodes are working:

```bash
❯ srun -N${NUM_NODES} hostname
```

### Setup Python Environment

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv /shared/venv/
source /shared/venv/bin/activate
pip install wheel
echo 'source /shared/venv/bin/activate' >> ~/.bashrc
```

or with conda:

```bash
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH="</path/to/miniconda3>/bin:$PATH"
```

To add miniconda to module:

```bash
cd /usr/share/modules/modulefiles
```

```bash
sudo nano miniconda3
```

Add these to nano

```bash
#%Module1.0
proc ModulesHelp { } {
    puts stderr "Adds Miniconda3 to your environment variables."
}
module-whatis "Loads Miniconda3 Environment."
set root /home/ubuntu/miniconda3
prepend-path PATH $root/bin
```

```bash
module use /usr/share/modules/modulefiles
```

## Sample Run

```bash
conda create -n ddp python=3.8
pip install torch torchvision numpy
```

```bash
cd /
sudo mkdir shared
sudo git clone https://github.com/PrincetonUniversity/multi_gpu_training.git
```

```slurm
#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=15G                # total memory per node (match to the instance type)
#SBATCH --gres=gpu:1             # number of allocated GPUs per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --output=job_%j.out      # Specify the file name for standard output
#SBATCH --error=job_%j.err       # Specify the file name for standard error

# Find a free port
export MASTER_PORT=$(shuf -i 6000-9999 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

# Define master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Load modules and activate environment
module purge
module load miniconda3
conda activate ddp

# Run the distributed PyTorch program
srun python simple_dpp.py
```

## Remember The Logs Are In Compute Nodes

You can check the logs in the compute nodes. For example, you can ssh into the
compute node and check the logs.

## Remember to Use Shared File System

Put your code or data in the shared file system, if not your compute nodes will
not be able to access them and have a copy of what you installed in the head
node.

Thinks like FSx, EFS, or S3 can be used as shared file system but for a poor
man's way and lazy way we can use `/home` as shared file system. Do not do this
in production. Remember to change to something like `777` for the shared
directory so that all nodes can access it.

```bash
sudo chmod -R 777 /home/multi_gpu_training
```

You can use this template to get FSx
(https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/slurm/config.yaml.template).

## Powering Down

```bash
aws ec2 stop-instances --instance-ids i-1234567890abcdef0 i-abcdef1234567890
```

## Troubleshooting

Usually at this stage if you face slurm creation failure, we need to inspect the
logs. Usually during creation one can ssh into head node, and you can list out
`ls /var/log` to see a wide range of logs. Or you can just follow
[AWS's documentation](https://docs.aws.amazon.com/parallelcluster/latest/ug/troubleshooting-v3-scaling-issues.html#troubleshooting-v3-key-logs)
to see what logs they recommend. They have a comprehensive
[guide](https://docs.aws.amazon.com/parallelcluster/latest/ug/troubleshooting-fc-v3-create-cluster.html)
on troubleshooting.

1. Check the system log from the head node (can be done in the AWS console):

    ```bash
    aws ec2 get-console-output --instance-id <instance-id> --region ap-southeast-1 --output text
    ```

2. You can also check relevant logs like:

    ```bash
    cat /var/log/cloud-init.log
    cat /var/log/cloud-init-output.log
    cat /var/log/cfn-init.log
    ```

See a sample log that an user reported
[here](https://github.com/aws/aws-parallelcluster/issues/5690).

### failureCode is HeadNodeBootstrapFailure with failureReason Cluster creation timed out

For example, I faced the
`failureCode is HeadNodeBootstrapFailure with failureReason Cluster creation timed out`
but AWS has good guide to troubleshoot
[here](https://docs.aws.amazon.com/parallelcluster/latest/ug/troubleshooting-fc-v3-create-cluster.html#create-cluster-head-node-bootstrap-timeout-failure-v3)

## Some Useful Commands

### Slurm Commands

```bash
sinfo # show information about nodes
squeue # show the queue
```

Useful command to check if the nodes are working and returning the hostname:

```bash
srun -N2 hostname # run a command on 2 nodes to see if they are working, returns the hostname
```

#### How Many GPUs?

```bash
scontrol show nodes # show more detailed information about nodes
sinfo -N -o "%N %G" # show the number of GPUs
```

### Delete Cluster

Assuming no other things like FSx, EFS, or S3 are attached to the cluster, you
can run:

```bash
pcluster delete-cluster --cluster-name <cluster-name> --region ap-southeast-1
```

and verify deletion:

```bash
pcluster list-clusters --region ap-southeast-1
```

You also need to delete your lingering VPCs, subnets etc:

```bash
aws ec2 delete-vpc --vpc-id [VPC-ID] --region ap-southeast-1
```

### Find the Instance ID

```bash
aws ec2 describe-instances --query "Reservations[*].Instances[*].{InstanceID: InstanceId, PublicDNS: PublicDnsName, State: State.Name, Tags: Tags}" --output table
```

If `table` format gives error like `list index out of range`, you can replace
`table` with `json` to slowly filter out.

### Find the Public DNS

```bash
aws ec2 describe-instances --instance-ids <instance-id> --query "Reservations[*].Instances[*].PublicDnsName" --output text
```

where the output of the previous command is the `<public-dns>`.

```bash
ssh -i </path/to/your-key.pem> <username>@<public-dns>
```

For `username` it is default to `ubuntu` for Ubuntu AMIs.

## References

-   https://www.hpcworkshops.com/index.html
-   https://qywu.github.io/2020/12/09/aws-slumr-pytorch.html
-   https://aws-parallelcluster.readthedocs.io/en/latest/configuration.html
-   https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/slurm/config.yaml.template
-   https://github.com/PrincetonUniversity/multi_gpu_training/tree/main
