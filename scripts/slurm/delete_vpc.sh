#!/bin/bash

# Set variables
CLUSTER_NAME="xxx"
REGION="xxx"
SUBNET_ID="xxx" # find from config.yaml
VPC_ID="xxx" # aws ec2 describe-subnets --subnet-ids "subnet-02cc9a3a21eecdc77" --query 'Subnets[0].VpcId' --output text # get the VPC ID

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
sleep 20

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
sleep 20

echo "Verifying subnet deletion..."
aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID"

# Delete route tables
echo "Deleting route tables..."
aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID" --query 'RouteTables[?Associations==`[]`].RouteTableId' --output text | xargs -n 1 -I {} aws ec2 delete-route-table --route-table-id {}
sleep 10

# Delete internet gateway
echo "Deleting internet gateway..."
IGW_ID=$(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$VPC_ID" --query 'InternetGateways[*].InternetGatewayId' --output text)
aws ec2 detach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID
aws ec2 delete-internet-gateway --internet-gateway-id $IGW_ID

# Delete the VPC
echo "Deleting VPC..."
aws ec2 delete-vpc --vpc-id $VPC_ID
sleep 20
echo "Verifying VPC deletion..."
aws ec2 describe-vpcs --vpc-ids $VPC_ID

echo "All resources have been deleted successfully."