# actor-sentiment-analysis

This project uses different sentiment analysis techniques and tools to perform sentiment classification on actors in a review. This is a fine-grained task that determines the reviewer's polarity opinion, positive negative or neutral, towards each actor mentioned in the review. We implement a lexicon look-up model, a recurrent neural network (RNN), a convolutional neural network (CNN) and an attention-based long-short term memory model (AT-LSTM).


Run the following command to get the sentiment of all actors in a given review file:

python3 NLPProject.py <inputfile> -l/r/c/a

<inputfile> must be the name of a text file inside the data/examples folder.
-l for lexicon
-r for RNN
-c for CNN
-a for AT-LSTM

Uses the SentiWordNet sentiment lexicon for lexicon-based classification.

The deep neural networks use the following implementation: https://github.com/awjuliani/DNN-Sentiment.

The AT-LSTM uses the following implementation: https://github.com/scaufengyang/TD-LSTM, based on the paper "Attention-based LSTM for Aspect-level Sentiment Classification" by Wang et al.

All data in data/Turk_Results obtained using Amazon Mechanical Turk workers.



Run NLPParsing.py as main to parse and output all review data from data/Turk_Results.csv.

Run LexiconSentimentAnalysis.py as main to compute lexicon-based analysis results on test data.

Run ReadDNNResults.py to read the results of the CNN/RNN classifiers on test data.
	Trained RNN/CNN models stored in lib/DNN-Sentiment/rnn_runs and lib/DNN-Sentiment/runs respectively.

Run lib/LSTM/reviews/actorReview.py to run AT-LSTM on review.
	Trained AT-LSTM model is too large for github, is available if requested.