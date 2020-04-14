# Sarcasm Detection

This project seeks to train a model for detecting sarcasm in a variety of contexts. The data used comes from the venerable Self-Annotated Reddit Corpus (SARC) from Princeton University <sup>1</sup>. Specifcally, I pull from the "train-unbalanced.csv.bz2"<sup>2</sup> dataset. The beauty of SARC is that it takes advantage of Redditers' tendency to self-label sarcastic comments with a '\s', eliminating the necessity of initial labeling on our part. Furthermore, Reddit comment threads are tree-like, providing ample context for sarcastic and non-sarcastic comments. Finally, Reddit is a platform for nearly any conceivable topic of conversation, providing NLP researchers with a rich variety of samples.   

## Preprocessing

The aforementioned dataset contains approximately 187M Reddit comments and their respective parent comments, pulled from tens of thousands of different subreddits. Of these, only a small percentage (< .01%) are labeled as sarcastic.  

I begin by loading the entire dataset into Google Cloud Storage. Then using my own spark-on-gcp app <sup>3</sup>, I remove all SARC features other than the content of the comment, the comment of the parent comment, the subreddit name, and the label. 

I retain only comments >= three words and <= fifty words (arbitrarily chosen) and then concatenate the parent comment and the child comment into one long sequence of text. Any null values are removed, and the resulting Spark dataframe is returned to Google Cloud Storage

## The Model

I use Google's Bidirectional Encoder Representations from Transformers' (BERT)<sup>4</sup> 12-layer, 768-hidden, 12-head model, downloaded from TensorFlow Hub, and unfreeze (i.e. fine-tune) the top two layers.  I then run the samples through a dense layer with relu activation, a dropout layer, and a dense layer with sigmoid activation. Loss is computed via binary cross entropy. 

To address sample imbalance, I first take all of the sarcastic-labeled comments and then take a sample from the non-sarcastic-labeled comments of equal size to that of the sarcastic-labeled comments. This re-balanced Spark dataframe is then tokenized via the BERT tokenizer and the requisite BERT input masks and segment_id masks are run through the TensorFlow graph. 

Currently, max sequence length is 128, batch size is set to 13, the number of epochs per sample Spark dataframe is 5, and the validation split is 0.1.

## Preliminary Results

I have yet to train for more than 30 minutes,but the model has so far achieved an out-of-sample accuracy of nearing 80 percent after 4 epochs, training on 2 GPUs. 

## To-Do List
•Train for longer across more sample Spark dataframes

•Try more complex BERT models

•Unfreeze more/fewer BERT layers

•Add additional LSTM and/or Attention layer on top

•Re-label instances for which the model's prediction was the most uncertain

## Citations

<sup>1</sup> Mikhail Khodak and Nikunj Saunshi and Kiran Vodrahalli (2017). A Large Self-Annotated Corpus for Sarcasm, https://arxiv.org/abs/1704.05579.

<sup>2</sup> https://nlp.cs.princeton.edu/SARC/1.0/main/

<sup>3</sup> https://github.com/chrisolen1/spark-on-gcp

<sup>4</sup> Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
