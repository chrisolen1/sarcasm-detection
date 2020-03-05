kubectl run $DRIVER_NAME --rm=true -ti -n $SPARK_NAMESPACE --image=$DRIVER_IMAGE --serviceaccount=$DRIVER_POD_SA
	
