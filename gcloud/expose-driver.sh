# Expose Spark driver internally
kubectl expose deployment $DRIVER_NAME --port=$DRIVER_PORT --type=ClusterIP --cluster-ip=None -n $SPARK_NAMESPACE

# Expose Jupyter service externally
kubectl expose deployment $DRIVER_NAME --port=8888 --type=LoadBalancer -n=$SPARK_NAMESPACE --name=jupyter
