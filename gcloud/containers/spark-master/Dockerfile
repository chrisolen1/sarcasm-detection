FROM brainlounge/jumppod:ubuntu-18.04

ENV PYTHON_VERSION=3.7
ENV SPARK_VERSION=2.4.4
ENV HADOOP_VERSION=2.7

ENV NAMESPACE=spark
ENV SA=spark-driver-sa
ENV K8S_CACERT=/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
ENV K8S_TOKEN=/var/run/secrets/kubernetes.io/serviceaccount/token
ENV DOCKER_IMAGE=gcr.io/sarc-1234/brainlounge/spark@sha256:ed1417e2136aef4d7b0abf52470ba043f7f7ec4a19a4ca1bb3650994e3f041a8
ENV DRIVER_NAME=spark-driver
ENV DRIVER_PORT=29413
ENV SPARK_HOME=/opt/spark

# Linux and Kubernetes dependencies
RUN apt-get update && apt-get install -y software-properties-common apt-transport-https apt-utils && \
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | tee -a /etc/apt/sources.list.d/kubernetes.list
     
# Kubectl
RUN apt-get update && apt-get install -y kubectl

# Java
RUN apt install -y openjdk-8-jdk && \
    cd opt && \
    wget http://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ./spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Python
COPY requirements.yaml /
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python$PYTHON_VERSION python3-pip && \
    python3.7 -m pip install -r /requirements.yaml
COPY start-jupyter.sh /
COPY data-cleaning.ipynb /

# GCS configs
RUN rm $SPARK_HOME/jars/guava-14.0.1.jar
ADD https://repo1.maven.org/maven2/com/google/guava/guava/23.0/guava-23.0.jar $SPARK_HOME/jars
ADD https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-latest-hadoop2.jar $SPARK_HOME/jars
RUN mkdir -p /opt/hadoop/conf
ADD https://github.com/GoogleCloudPlatform/spark-on-k8s-gcp-examples/blob/master/conf/core-site.xml /opt/hadoop/conf
ADD https://github.com/GoogleCloudPlatform/spark-on-k8s-gcp-examples/blob/master/conf/spark-env.sh $SPARK_HOME/conf
RUN mkdir /mnt/secrets
COPY sarc-bucket-sa.json /mnt/secrets
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin
