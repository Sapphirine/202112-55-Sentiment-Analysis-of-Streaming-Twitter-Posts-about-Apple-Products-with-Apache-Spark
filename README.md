## Name: Ross Koval (CVN Student)

## UNI: rk3091

## Course: EECS6893

## Final Project: Large-Scale Sentiment Analysis of Streaming Twitter Posts about Apple Products with Apache Spark

# Introduction

In this project, I performed large-scale sentiment analysis of streaming Twitter Posts about Apple Consumer Products (e.g. iPhone, iPad, MacBook, etc.). First, I experimented with and trained a state-of-the-art model on the massive Sentiment140 Labeled Twitter Posts dataset using both traditional Machine Learning and more modern Deep Learning methods with Spark ML and Spark NLP. Next, I applied this trained model to Streaming Twitter Posts, using Tweepy and Spark Streaming, to track the real-time sentiment of Twitter users towards Apple Consumer Products over time and test if this sentiment is correlated with changes in Apple Stock Price. The system is able to measure and track this sentiment and record the Apple Stock Price in 1-minute increments, and records the results to a data bucket on Google Cloud Storage. Then, I visualized the results and identified trends in user sentiment towards each Apple product and tested if they are correlated with changes in Apple Stock Price by performing a number of regressions over different time horizons across Hashtags.

# Environment 

The Streaming and Training code was performed with Spark on a GCP Dataproc Cluster. Please note that the cluster was instantiated with the following command to allow for compatability with Spark NLP package and install the necessary jar files on each worker node. Pleas sure the following link for more information.

https://nlp.johnsnowlabs.com/docs/en/install#gcp-dataproc-support

"""

gcloud dataproc clusters create final-project2 --region=us-east1 --image-version=2.0 --master-machine-type=n1-standard-4 --worker-machine-type=n1-standard-2 --master-boot-disk-size=128GB --worker-boot-disk-size=128GB --num-workers=2 --bucket=eecs6893-hw2-cluster --optional-components=JUPYTER --enable-component-gateway --metadata “PIP_PACKAGES=spark-nlp spark-nlp-display google-cloud-bigquery google-cloud-storage requests_oauthlib tweepy==3.10.0” --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh --properties spark:spark.serializer=org.apache.spark.serializer.KryoSerializer,spark:spark.driver.maxResultSize=0,spark:spark.kryoserializer.buffer.max=2000M,spark:spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.2

"""

# Code Files

The code for this project is contained in the following set of Python Code Files and Jupyter Notebooks, described below.

1) twitter_stream.ipynb 

This code, from the HW2 Starter Code, uses Tweepy and Spark Streaming to stream real-time Tweets containing the specified hashtags about Apple Products and direct them through the specified port to be collected. 

2) stream_engine.py

This code will accept the streaming Tweets from twitter_stream.py in 1-minute batch intervals, preprocess them and store to Google Cloud Storage Bucket for offline testing, load the pretrained BERT model, perform online sentiment analysis of them, and store the processed analytics grouped by the hashtag to the GCP Storage Bucket. 

3) stream_stock_prices.ipynb

This code will query Yahoo Finance for the most recent market price of Apple Stock (ticker: AAPL) using the yfinance Python API, and store the timestamped results to a GCP BigQuery table in batch intervals of 1-minute for a total of 8 hours per day. 

4) model_training_spark.ipynb

This Jupyter Notebook contains the code that loads the Sentiment140 labeled Tweets dataset from TensorFlow datasets, preprocesses the text, splits the data into train and test sets, trains a variety of ML models using Spark ML and Spark NLP, and evaluates their performance on the test set. 

5) model_training_tf.ipynb

This Jupyter Notebook contains the code that finetunes the pretrained BERT model using TensorFlow and Huggingface, and saves the pretrained model in a Spark NLP compatible format that allows it be applied in a pipeline downstream for both online and offline testing. 

6) offline_data_exploration.ipynb

This Jupyter Notebook contains the code that performs exploratory data analysis on the offline Sentiment140 dataset to better understand it by computing descriptive statistics, vocabulary frequency counts, etc. 

7) online_data_exploration_and_results.ipynb

This Jupyter Notebook contains the code that performs exploratory data analysis on the online Streaming Twitter Posts and Apple Stock Prices datasets to better understand them by computing descriptive statistics, vocabulary frequency counts. It also contains all code used to estimate a number of regressions over different aggregation frequencies and time horizons at the Hashtag level and use Exponential Smoothing, and produce the regression results.


# References


## Tutorials

HW2 Starter Code for Twitter Streaming and Processing

https://spark.apache.org/docs/latest/ml-features.html

https://spark.apache.org/docs/latest/ml-pipeline.html

https://spark.apache.org/docs/latest/ml-classification-regression.html

https://www.geeksforgeeks.org/generating-word-cloud-python/

https://stackoverflow.com/questions/38839924/how-to-combine-n-grams-into-one-vocabulary-in-spark

https://nlp.johnsnowlabs.com/docs/en/install#gcp-dataproc-support

https://nlp.johnsnowlabs.com/docs/en/transformers#bertsentenceembeddings

https://nlp.johnsnowlabs.com/docs/en/annotators

https://nlp.johnsnowlabs.com/docs/en/training

https://nlp.johnsnowlabs.com/docs/en/transformers

https://spark.apache.org/docs/latest/streaming-programming-guide.html

https://developer.twitter.com/en

https://pypi.org/project/yfinance/



## Academic Papers

[1]	https://en.wikipedia.org/wiki/Tf%E2%80%93idf

[2]	Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In Advances in neural information processing systems, pp. 5998-6008. 2017.

[3]	Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

[4]	Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).

[5]	Sanh, Victor, Lysandre Debut, Julien Chaumond, and Thomas Wolf. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).

[6]	https://www.tensorflow.org/datasets/catalog/sentiment140

[7]	Go, Alec, Richa Bhayani, and Lei Huang. "Twitter sentiment classification using distant supervision." CS224N project report, Stanford 1, no. 12 (2009): 2009.

[8]	McCloskey, Michael, and Neal J. Cohen. "Catastrophic interference in connectionist networks: The sequential learning problem." In Psychology of learning and motivation, vol. 24, pp. 109-165. Academic Press, 1989.

[9]	Mohammad, Saif M., Svetlana Kiritchenko, and Xiaodan Zhu. "NRC-Canada: Building the state-of-the-art in sentiment analysis of tweets." arXiv preprint arXiv:1308.6242 (2013).
[10]	https://www.statista.com/statistics/382260/segments-share-revenue-of-apple/

[11]	Newey, W. K., & West, K. D. (1987). Hypothesis Testing with Efficient Method of Moments Estimation. International Economic Review, 28(3), 777–787. https://doi.org/10.2307/2526578

[12]	https://en.wikipedia.org/wiki/Exponential_smoothing

[13] 	RoBERTa: A Robustly Optimized BERT Pretraining Approach - https://arxiv.org/abs/1907.11692

[14] 	Universal Language Model Fine-tuning for Text Classification - https://arxiv.org/abs/1801.06146

[15]	Universal Sentence Encoder - https://arxiv.org/abs/1803.11175 





