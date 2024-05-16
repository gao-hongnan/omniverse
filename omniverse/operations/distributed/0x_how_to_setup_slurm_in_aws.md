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
ParallelCluster.

Note that the version of AWS that works as of writing is below:

```bash
awscli                              1.32.93
aws-parallelcluster                 3.9.1
```

Some breaking updates may render current commands invalid.

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
❯ pcluster create-cluster --cluster-configuration config.yaml --cluster-name <YOUR-CLUSTER-NAME> --region <REGION>
```

This command specifies the cluster configuration file (`config.yaml`), the name
of the cluster, and the AWS region where you want the cluster to be deployed.

### 3. Monitor the Cluster Creation

The cluster creation process may take some time. You can monitor the progress by
running the following command:

```bash
❯ pcluster describe-cluster --cluster-name <YOUR-CLUSTER-NAME> --region ap-southeast-1
```

## Shared File System

Put your code or data in the shared file system, if not your compute nodes will
not be able to access them and have a copy of what you installed in the head
node. This means if you need to write or read data from the compute nodes, you
need to put them in the shared file system.

Thinks like FSx, EFS, or S3 can be used as shared file system but for a poor
man's way and lazy way we can use `/home` as shared file system. Do not do this
in production. Remember to change to something like `777` for the shared
directory so that all nodes can access it.

```bash
❯ sudo chmod -R 777 /home/multi_gpu_training
```

Alternatively, you can also mount an EFS volume to the head node and then share
it with the compute nodes. For example, you can add the below configuration to
the `config.yaml` file:

```yaml
SharedStorage:
    - MountDir: /shared
      Name: my-efs
      StorageType: Efs
      EfsSettings:
          PerformanceMode: generalPurpose
          ThroughputMode: bursting
```

EFS is cheap and can be used as a shared file system but if you want things like
FSx, which is for high performance computing, you can refer to a sample
[template from PyTorch](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/slurm/config.yaml.template).

## Logging into the Head Node

During creation of the cluster, you can ssh into the head node using the
following command:

```bash
❯ pcluster ssh --cluster-name <YOUR-CLUSTER-NAME> -i <path-to-your-key.pem> --region <REGION>
```

Alternatively you can use normal ssh to login into the head node as well which
is useful if you are using vscode remote to login. First, get the public DNS of
the head node (you can just get the instance id from the AWS console and then
get the public DNS):

```bash
❯ aws ec2 describe-instances --instance-ids <INSTANCE-ID> --query "Reservations[*].Instances[*].PublicDnsName" --output text
```

where the output of the previous command is the `<public-dns>`.

Then the ssh command is:

```bash
❯ ssh -i </path/to/your-key.pem> <username>@<public-dns>
```

For `username` it defaults to `ubuntu` for Ubuntu AMIs.

### Compute Node

Due to quota limit, our compute node is `g4dn.xlarge` and has only 1 GPU per
node.

### SLURM Status

We can check status of the slurm cluster:

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

#### Virtual Environment

```bash
#!/usr/bin/env sh
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv /shared/venv/
source /shared/venv/bin/activate
pip install wheel
echo 'source /shared/venv/bin/activate' >> ~/.bashrc
```

#### Miniconda

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

And if you don't use it in script, you can just do the following.

```bash
source /etc/profile.d/modules.sh
module use /usr/share/modules/modulefiles
module load miniconda3
conda create -n ddp python=3.9
conda init
conda activate ddp
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

## Delete Cluster

### Delete ParallelCluster

```bash
pcluster delete-cluster --cluster-name <YOUR-CLUSTER-NAME> --region <REGION>
```

and verify deletion:

```bash
pcluster list-clusters --region <REGION>
```

### Delete Network Resources

You also need to delete your lingering VPCs, subnets etc. First, we idenfity all
the network resources associated with the cluster including VPC, subnets etc. Go
[here](https://ap-southeast-1.console.aws.amazon.com/vpcconsole/home?region=ap-southeast-1#Home:)
to see existing resources.

First get your subnet id from `config.yaml`.

Now we first delete NAT.

```bash
aws ec2 describe-subnets --subnet-ids <subnet-XXX> --query 'Subnets[0].VpcId' --output text # get the VPC ID
aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=<vpc-XXX>" # get nat id in the form of nat-xxx
aws ec2 delete-nat-gateway --nat-gateway-id <nat-XXX> # delete the nat gateway
aws ec2 describe-nat-gateways --nat-gateway-ids <nat-XXX> # check if it is deleted
```

Then we detach and delete network interfaces.

```bash
# Detach network interfaces
aws ec2 describe-network-interfaces \
    --filters "Name=vpc-id,Values=<vpc-XXX>" \
    --query 'NetworkInterfaces[*].[NetworkInterfaceId,Attachment.AttachmentId]' \
    --output text | while read -r interface_id attachment_id; do
      if [ ! -z "$attachment_id" ]; then
        aws ec2 detach-network-interface --attachment-id $attachment_id
      fi
    done

# Delete network interfaces
aws ec2 describe-network-interfaces \
    --filters "Name=vpc-id,Values=<vpc-XXX>" \
    --query 'NetworkInterfaces[*].NetworkInterfaceId' \
    --output text | xargs -I {} aws ec2 delete-network-interface --network-interface-id {}

# Run again to see if deleted or not
aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=<vpc-XXX>"
```

Next, delete subnets

```bash
aws ec2 describe-subnets --filters "Name=vpc-id,Values=<vpc-XXX>" --query 'Subnets[*].SubnetId' --output text | xargs -n 1 -I {} aws ec2 delete-subnet --subnet-id {}
# Check if deleted
aws ec2 describe-subnets --filters "Name=vpc-id,Values=<vpc-XXX>" --query 'Subnets[*].SubnetId' --output text
```

Next, delete route tables

```bash
aws ec2 describe-route-tables --filters "Name=vpc-id,Values=<vpc-XXX>" --query 'RouteTables[?Associations==`[]`].RouteTableId' --output text | xargs -n 1 -I {} aws ec2 delete-route-table --route-table-id {}
```

Now delete the internet gateway

```bash
# list the internet gateways
aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=<vpc-XXX>" --query 'InternetGateways[*].InternetGatewayId' --output text

aws ec2 detach-internet-gateway --internet-gateway-id <igw-XXX> --vpc-id <vpc-XXX>
aws ec2 delete-internet-gateway --internet-gateway-id <igw-XXX>
```

Lastly, delete the VPC

```bash
aws ec2 delete-vpc --vpc-id <vpc-XXX>
# check if deleted
aws ec2 describe-vpcs --vpc-ids <vpc-XXX>
```

### Consolidated Script

```bash
#!/bin/bash

# Set variables
CLUSTER_NAME="<YOUR-CLUSTER-NAME>"
REGION="<REGION>"
VPC_ID="<vpc-XXX>"
SUBNET_ID="<subnet-XXX>"

# Delete ParallelCluster
echo "Deleting AWS ParallelCluster..."
pcluster delete-cluster --cluster-name $CLUSTER_NAME --region $REGION

# Wait and verify deletion
echo "Listing all clusters to verify deletion..."
pcluster list-clusters --region $REGION

# Describe NAT Gateway
echo "Fetching NAT Gateway ID..."
NAT_ID=$(aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=$VPC_ID" --query 'NatGateways[0].NatGatewayId' --output text)
echo "Deleting NAT Gateway..."
aws ec2 delete-nat-gateway --nat-gateway-id $NAT_ID
echo "Verifying NAT Gateway deletion..."
aws ec2 describe-nat-gateways --nat-gateway-ids $NAT_ID

# Detach and delete network interfaces
echo "Detaching and deleting network interfaces..."
aws ec2 describe-network-interfaces \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'NetworkInterfaces[*].[NetworkInterfaceId,Attachment.AttachmentId]' \
    --output text | while read -r interface_id attachment_id; do
      if [ ! -z "$attachment_id" ]; then
        aws ec2 detach-network-interface --attachment-id $attachment_id
      fi
      aws ec2 delete-network-interface --network-interface-id $interface_id
    done

# Delete subnets
echo "Deleting subnets..."
aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[*].SubnetId' --output text | xargs -n 1 -I {} aws ec2 delete-subnet --subnet-id {}
echo "Verifying subnet deletion..."
aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID"

# Delete route tables
echo "Deleting route tables..."
aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID" --query 'RouteTables[?Associations==`[]`].RouteTableId' --output text | xargs -n 1 -I {} aws ec2 delete-route-table --route-table-id {}

# Delete internet gateway
echo "Deleting internet gateway..."
IGW_ID=$(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$VPC_ID" --query 'InternetGateways[*].InternetGatewayId' --output text)
aws ec2 detach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID
aws ec2 delete-internet-gateway --internet-gateway-id $IGW_ID

# Delete the VPC
echo "Deleting VPC..."
aws ec2 delete-vpc --vpc-id $VPC_ID
echo "Verifying VPC deletion..."
aws ec2 describe-vpcs --vpc-ids $VPC_ID

echo "All resources have been deleted successfully."
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
    cat /var/log/chef-client.log
    cat slurmctld.log
    cat parallelcluster/clusterstatusmgtd
    ```

See a sample log that an user reported
[here](https://github.com/aws/aws-parallelcluster/issues/5690).

### failureCode is HeadNodeBootstrapFailure with failureReason Cluster creation timed out

For example, I faced the
`failureCode is HeadNodeBootstrapFailure with failureReason Cluster creation timed out`
but AWS has good guide to troubleshoot
[here](https://docs.aws.amazon.com/parallelcluster/latest/ug/troubleshooting-fc-v3-create-cluster.html#create-cluster-head-node-bootstrap-timeout-failure-v3).

Sometimes it is simply because your AWS account has not enough quotas. For me
the concrete error for this can be found in
`sudo cat /var/log/parallelcluster/clustermgtd.events`.

```bash
{"datetime": "2024-05-05T07:59:58.279+00:00", "version": 0, "scheduler": "slurm", "cluster-name": "distributed-training", "node-role": "HeadNode", "component": "clustermgtd", "level": "WARNING", "instance-id": "i-08d41f68bf12ca54b", "event-type": "node-launch-failure-count", "message": "Number of static nodes that failed to launch a backing instance after node maintenance", "detail": {"failure-type": "vcpu-limit-failures", "count": 1, "error-details": {"VcpuLimitExceeded": {"count": 1, "nodes": [{"name": "distributed-queue-st-g4dn12xlarge-2"}]}}}}
```

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

### Find the Instance ID

```bash
aws ec2 describe-instances --query "Reservations[*].Instances[*].{InstanceID: InstanceId, PublicDNS: PublicDnsName, State: State.Name, Tags: Tags}" --output table
```

If `table` format gives error like `list index out of range`, you can replace
`table` with `json` to slowly filter out.

### Stop EC2 Instances

```bash
aws ec2 stop-instances --instance-ids i-1234567890abcdef0 i-abcdef1234567890
```

## References

-   https://www.hpcworkshops.com/
-   https://qywu.github.io/2020/12/09/aws-slumr-pytorch.html
-   https://aws-parallelcluster.readthedocs.io/en/latest/configuration.html
-   https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/slurm/config.yaml.template
-   https://github.com/PrincetonUniversity/multi_gpu_training/tree/main
-   https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/slurm/setup_pcluster_slurm.md
-   https://aws.amazon.com/blogs/opensource/aws-parallelcluster/
