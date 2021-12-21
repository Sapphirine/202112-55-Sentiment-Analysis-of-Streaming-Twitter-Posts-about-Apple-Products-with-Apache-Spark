'''
Source: HW2 Starter Code
'''

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SQLContext
import sys
import requests
import time
import subprocess
import re
from google.cloud import bigquery
import datetime as dt
from pyspark.ml import PipelineModel

hashtags = ['aapl', 'macbook', 'iphone', 'ipad', 'imac', 'iwatch']
IP = 'localhost'
PORT = 9001


def save_to_gcs(df):
    """
    Save each RDD in this DStream to google storage
    Args:
        rdd: input rdd
        output_directory: output directory in google storage
        columns_name: columns name of dataframe
        mode: mode = "overwirte", overwirte the file
              mode = "append", append data to the end of file
    """
    output_directory = "gs://eecs6893-hw2-cluster/final-project/streaming_text/test/"
    if df.count() >= 1:
        df.coalesce(1).write.save(output_directory, format="csv", mode='append', header=True)


def clean_text(text):
    
    """This function will clean the raw Tweet text and only extract valid alphanumeric characters"""

    text = text.lower()
    text = text.replace("'", "")
    text = re.sub('@(\w+)', ' ', text)
    text = re.sub('\W+', ' ', text)
    text = re.sub('<[^<]+?>', ' ', text)
    text = re.sub('[^A-Za-z0-9\.\!\;\:\,\?]+', ' ', text)
    text = text.replace('amp', 'and').strip()

    return text


def save_rdd_to_gcs(t, rdd):
    
    """This function will save an RDD to a bucket on the GCP cluster if the RDD is not empty"""

    if rdd.count() >= 1:
        rdd = rdd.map(lambda x: clean_text(x))
        rdd = rdd.map(lambda x: Row(text=x))
        df = rdd.toDF()
        output_dir = "gs://eecs6893-hw2-cluster/final-project/raw_stream/"
        df.coalesce(1).write.save(output_dir + str(t), format="text", mode='append')
        print('Saving Raw Text File: %s' % str(t))


def load_bert_model(root_dir, model_name):
    
    
    """This function will load a pretrained BERT Model and format according to Spark NLP requirements into a Pipeline object"""

    model = BertForSequenceClassification.load(root_dir + "/{}_spark_nlp".format(model_name)) \
        .setInputCols(["document", 'token']) \
        .setOutputCol("Pred")

    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    sequenceClassifier = model \
        .setInputCols(["document", "token"]) \
        .setOutputCol("Pred") \
        .setCaseSensitive(False)

    pipeline_model = Pipeline().setStages([
        documentAssembler,
        tokenizer,
        sequenceClassifier
    ])

    return pipeline_model


def score_sentiment(rdd):

    """This function accepts an RDD, a Spark NLP Model, and a list of hashtags,
    and it performs inference on the RDD of Tweets, scoring the sentiment of each one,
    and aggregates the sentiment in each batch by the associated Hashtag contained in it
    with a simple average score"""

    if rdd.count() >= 1:

        print('RDD is not empty')
        root_dir = 'gs://eecs6893-hw2-cluster/final-project/training/'
        model_name = 'BERT_TEST_EPOCH_5'
        print("Loading Model", model_name)
        pipeline_model = load_bert_model(root_dir, model_name)
        print("Loaded Model", model_name)

        ts = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        MIN_WORDS = 3
        rdd = rdd.map(lambda x: clean_text(x))
        rowRDD = rdd.map(lambda x: Row(text=x))
        df = rowRDD.toDF()
        df = df.filter(size(split(df['text'], ' ')) >= MIN_WORDS)
        preds = pipeline_model.fit(df).transform(df)
        preds = preds.withColumn('PredLabel', substring(preds['Pred.result'].getItem(0), -1, 1).cast(DoubleType()))
        preds = preds.withColumn('Timestamp', lit(ts))

        hashtags = ['aapl', 'macbook', 'iphone', 'ipad', 'imac', 'iwatch']
        for hashtag in hashtags:
            ht = preds.filter(array_contains(split(preds['text'], ' '), hashtag))
            ht = ht.withColumn('Hashtag', lit(hashtag))
            ht = ht.select('Timestamp', 'Hashtag', 'PredLabel')
            save_to_gcs(ht)
            pdf = ht.toPandas()
            pdf.to_csv("gs://eecs6893-hw2-cluster/final-project/streaming_text/test/", mode='a', header=False, index=False)
            print('Writing Python DataFrame to Storage Bucket: %s' % ts)
            ht = ht.groupBy('Hashtag', 'Timestamp').mean('PredLabel')
            ht.write.format('com.google.cloud.spark.bigquery').option('table', 'final_project.tweets-sentiment').mode('append').save()
            print('Writing Spark DataFrame to BigQuery: %s' % ts)

    else:
        print('RDD is empty')
        print(rdd.count())


if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("TwitterStreamApp") \
        .master("local[2]") \
        .config("spark.driver.memory", "32G") \
        .config("spark.executor.memory", "32G") \
        .config("spark.driver.maxResultSize", "8G") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.2").getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    sql_context = SQLContext(sc)
    batch_interval = 1 * 60
    STREAMTIME = 8 * 60 * 60
    ssc = StreamingContext(sc, batch_interval)
    ssc.checkpoint("~/checkpoint_TwitterApp")

    dataStream = ssc.socketTextStream(IP, PORT)
    dataStream.pprint()

    dataStream.foreachRDD(lambda t, rdd: save_rdd_to_gcs(t, rdd))

    root_dir = 'gs://eecs6893-hw2-cluster/final-project/training/'
    model_name = 'BERT_TEST_EPOCH_5'
    pipeline_model = load_bert_model(root_dir, model_name)
    dataStream.foreachRDD(lambda rdd: score_sentiment(rdd, pipeline_model))

    ssc.start()
    time.sleep(STREAMTIME)
    ssc.awaitTermination(timeout=60 * 60)
    ssc.stop(stopSparkContext=False, stopGraceFully=True)

