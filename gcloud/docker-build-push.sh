cd containers/spark-master
echo $PWD
docker build -t ${DRIVER_IMAGE} .
echo "Successfully Built Driver Image"
docker push ${DRIVER_IMAGE}
echo "Successfully Pushed Driver Image"
cd ../spark-worker
echo $PWD
docker build -t ${WORKER_IMAGE} -f kubernetes/dockerfiles/spark/Dockerfile .
echo "Successfully Built Worker Image"
docker push ${WORKER_IMAGE}
echo "Successfully Pushed Worker Image"
