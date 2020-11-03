

ARG BASE_IMG=${BASE_IMG}
FROM ${BASE_IMG} 

ENV PATH="/opt/ml/code:${PATH}"
## install cmake 3.17 && opencv

WORKDIR /opt/program

COPY main.py /opt/program
COPY util.py /opt/program
#entry point for sagemaker training job 
ENTRYPOINT ["python", "main.py"]

