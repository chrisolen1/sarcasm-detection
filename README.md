# sarcasm_detection

This project seeks to train a model for detecting sarcasm in a variety of contexts. The data used comes from the venerable Self-Annotated Reddit Corpus (SARC) from Princeton University <sup>1</sup>. Specifcally, I pull from the "train-unbalanced.csv.bz2"<sup>2</sup> dataset. The beauty of SARC is that it takes advantage of Redditers' tendency to self-label sarcastic comments with a '\s', eliminating the necessity of initial labeling on our part. Furthermore, Reddit comment threads are tree-like, providing ample context for sarcastic and non-sarcastic comments. Finally, Reddit is a platform for nearly any conceivable topic of conversation, providing NLP researchers with a rich variety of samples.   

## methodology

The aforementioned dataset contains approximately 187M Reddit comments and their respective parent comments, pulled from tens of thousands of different subreddits. Of these, only a small percentage (< .01%) are labeled as sarcastic.  









<sup>1</sup> Mikhail Khodak and Nikunj Saunshi and Kiran Vodrahalli (2017). A Large Self-Annotated Corpus for Sarcasm, https://arxiv.org/abs/1704.05579.
<sup>2</sup> https://nlp.cs.princeton.edu/SARC/1.0/main/
