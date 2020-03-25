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

def tokenize_sample(context):
    
    """
    To be applied over Spark dataframe.
    Takes a string and converts it to token IDs via bert_tokenizer,
    adding the necessary beginning and end tokens

    Returns: Array of bert token ids for each row of Spark dataframe (requires udf)
    """
    
    tokenized = ["[CLS]"] + tokenizer.tokenize(context) + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(tokenized)
    
    return ids

def generate_epoch_df(sarcastic, non_sarcastic, ratio, n_epochs):
    
    """
    Generates a Pandas dataframe of equal label distribution over which 
    we can perform mini-batch gradient descent. Each generated df is
    to be iterator over multiple times during training
    """
    number = 0
    while number < n_epochs:
        non_sarc_samp = non_sarcastic.sample(ratio) # making label dist equal
        
        # combine sampled non_sarcastic and whole sarcastic
        epoch_df = sarcastic.union(non_sarc_samp)
        
        # tokenize context column via spark udf
        tokenize_sample_udf = F.udf(tokenize_sample, ArrayType(IntegerType()))
        epoch_df = epoch_df.withColumn("tokens", tokenize_sample_udf(epoch_df.context))
        
        # split into X and y numpy arrays
        X = np.array(epoch_df.select('tokens').collect())
        y = np.array(epoch_df.select('label').collect())
        
        # yield one call at a time
        yield X, y
        number += 1





