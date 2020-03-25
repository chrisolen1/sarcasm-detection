import pyspark
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

import tensorflow as tf
import tensorflow_hub as thub
import bert

import pandas as pd
import numpy as np

import re

import random

import os
from tqdm import tqdm

def init_bert(bert_version="https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/1", trainable=True, do_lower_case=False):

	"""
	Load params and vocab from existing BERT model

	Returns: bert_layer object and bert_tokenizer
	"""

	bert_layer = thub.KerasLayer(bert_version,trainable=trainable)
	vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
	BertTokenizer = bert.bert_tokenization.FullTokenizer
	tokenizer = BertTokenizer(vocabulary_file, do_lower_case=do_lower_case)

	return bert_layer, tokenizer


def init_spark(dynamic_allocation="True", executor_cores="2"):

	"""
	Initialize Spark context

	Returns: Spark session object
	"""

	config = pyspark.SparkConf().setAll([("spark.dynamicAllocation.enabled",dynamic_allocation),
                                    ("spark.executor.cores",executor_cores)])
	sc = SparkContext(conf=config)
	spark = SparkSession(sc)

	return sc, spark

def load_data(spark_context, bucket_name, dataset):
    
    """
    Takes in the name of csv file in GCS and the name of the GCS bucket
    and loads it in as a Spark dataframe

    Returns: Spark df of only sarcastic, Spark df of only non-sacrastic, and the ratio 
    between the two of them 
    """
    
    sarc = spark_context.read.csv("gs://{}/{}.csv".format(bucket_name, dataset), 
                          inferSchema=True, header=False, sep = ',')
    sarc = sarc.withColumnRenamed('_c0','label').withColumnRenamed('_c1','subreddit').withColumnRenamed('_c2','context')
    sarcastic = sarc.where(F.col('label')==1)
    non_sarcastic = sarc.where(F.col('label')==0)
    sarc_cnt = sarcastic.count()
    non_sarc_cnt = non_sarcastic.count()
    ratio = sarc_cnt / non_sarc_cnt
    
    # dropping subreddit column before returning
    sarcastic = sarcastic.drop('subreddit')
    non_sarcastic = non_sarcastic.drop('subreddit')
    
    return sarcastic, non_sarcastic, ratio







