#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM openjdk:8-jdk-slim

ARG spark_jars=jars
ARG img_path=kubernetes/dockerfiles
ARG k8s_tests=kubernetes/tests

ENV PYTHON_VERSION=3.7

# Before building the docker image, first build and make a Spark distribution following
# the instructions in http://spark.apache.org/docs/latest/building-spark.html.
# If this docker file is being used in the context of building your images from a Spark
# distribution, the docker build command should be invoked from the top level directory
# of the Spark distribution. E.g.:
# docker build -t spark:latest -f kubernetes/dockerfiles/spark/Dockerfile .

# Linux
RUN set -ex && \
    apt-get update && \
    ln -s /lib /lib64 && \
    apt-get install -y software-properties-common curl bash tini libc6 libpam-modules krb5-user libnss3 && \
    mv /usr/bin/tini /sbin && \ 
    mkdir -p /opt/spark && \
    mkdir -p /opt/spark/work-dir && \
    touch /opt/spark/RELEASE && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd

# Spark
COPY ${spark_jars} /opt/spark/jars
COPY bin /opt/spark/bin
COPY sbin /opt/spark/sbin
COPY ${img_path}/spark/entrypoint.sh /opt/
COPY examples /opt/spark/examples
COPY ${k8s_tests} /opt/spark/tests
COPY data /opt/spark/data

ENV SPARK_HOME=/opt/spark

# Python
COPY requirements.yaml /
RUN apt-get update && apt-get install -y python$PYTHON_VERSION python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3.7 -m pip install -r /requirements.yaml

ENV PYTHONPATH={SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-*.zip:${SPARK_HOME}/python
ENV PYSPARK_PYTHON=python3

# GCS configs
RUN rm $SPARK_HOME/jars/guava-14.0.1.jar
ADD https://repo1.maven.org/maven2/com/google/guava/guava/23.0/guava-23.0.jar $SPARK_HOME/jars
ADD https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-latest-hadoop2.jar $SPARK_HOME/jars
RUN mkdir -p /opt/hadoop/conf
ADD https://github.com/GoogleCloudPlatform/spark-on-k8s-gcp-examples/blob/master/conf/core-site.xml /opt/hadoop/conf
RUN mkdir -p $SPARK_HOME/conf
ADD https://github.com/GoogleCloudPlatform/spark-on-k8s-gcp-examples/blob/master/conf/spark-env.sh $SPARK_HOME/conf
RUN mkdir /mnt/secrets
COPY sarc-bucket-sa.json /mnt/secrets 
#RUN curl -sSL https://sdk.cloud.google.com | bash
#ENV PATH $PATH:/root/google-cloud-sdk/bin

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]
