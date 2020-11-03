#!/bin/bash

# The name of our algorithm
algorithm_name=vgg16-tl-tf

cd tf22-container 

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}
echo $region


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 763104351884 --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
base_img='763104351884.dkr.ecr.'$region'.amazonaws.com/tensorflow-training:2.2.0-gpu-py37-cu101-ubuntu18.04'
echo 'base_img:'$base_img

docker build  -t ${algorithm_name} -f Dockerfile  --build-arg BASE_IMG="${base_img}" .  
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}
