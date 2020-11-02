

ARG BASE_IMG=${BASE_IMG}
FROM ${BASE_IMG} 

ENV PATH="/opt/ml/code:${PATH}"

## install cmake 3.17 && opencv
RUN apt-get remove cmake
RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-unauthenticated \
    python3-opencv \
    libopencv-dev \
    libopencv-contrib-dev
RUN curl -sSL https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Linux-x86_64.tar.gz | tar -xzC /opt
RUN mv /opt/cmake-3.17.2-Linux-x86_64 /opt/cmake
RUN ln -s /opt/cmake/bin/cmake /usr/bin/cmake

## fix /usr/local/cuda-10.0/compat/libcuda.so
RUN bash -c 'echo "/usr/local/cuda-10.0/compat" > /etc/ld.so.conf.d/cuda.conf'
RUN ldconfig -v

## Build darknet
RUN cd opt && git clone https://github.com/AlexeyAB/darknet
RUN cd /opt/darknet && sed -i 's/GPU=0/GPU=1/g' Makefile && \
    sed -i 's/CUDNN=0/CUDNN=1/g' Makefile && \
    sed -i 's/LIBSO=0/LIBSO=1/g' Makefile
RUN cd /opt/darknet && ./build.sh

ENV PATH="/opt/darknet:${PATH}"
WORKDIR /opt/program

COPY main.py /opt/program
#entry point for sagemaker training job 
ENTRYPOINT ["python", "main.py"]

