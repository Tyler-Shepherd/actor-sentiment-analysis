import csv
from pprint import pprint
import NLPParsing
import math


# ../../../../../Anaconda3/python getSentiment.py actor_sentence RNN 1524617308



def Read_CSV(csv_file):
	sentence_to_sentiment = {}
	counter = 0

	sentiments = []

	# Open the results of the CNN sentiment analyzer
	with open(csv_file, 'r') as csvfile:
		spamreader = csv.reader(csvfile)
		for row in spamreader:
			counter += 1
			if counter == 1:
				continue
			sentence = row[0]
			sentiment = row[2]
			sentence_to_sentiment[sentence] = float(sentiment)
			sentiments.append(float(sentiment))

	num_sentiments = len(sentiments)
	sum_sentiments = sum(sentiments)
	sentiment_mean = float(sum_sentiments) / num_sentiments
	stdev = sum([(a - sentiment_mean) ** 2 for a in sentiments])
	stdev = stdev / num_sentiments
	stdev = math.sqrt(stdev)

	# print("mean", sentiment_mean)
	# print("stdev", stdev)

	return sentence_to_sentiment, sentiment_mean, stdev




def GetSentiment(csv_file, actor_sentences):
	sentence_to_sentiment, mean, stdev = Read_CSV(csv_file)

	pred_sentiment = 0
	num_sentences = len(actor_sentences)

	for sentence in sentence_to_sentiment.keys():
		for actor_sentence in actor_sentences:
			sentence_norm = "".join(sentence.lower().split(" "))
			actor_sentence_norm = "".join(actor_sentence.lower().split(" "))

			# DNN-Sentiment splits sentences weirdly so we need to check if every DNN-Sentiment result sentence is part of any sentence the actor is in
			if sentence_norm in actor_sentence_norm or actor_sentence_norm in sentence_norm:
				pred_sentiment += sentence_to_sentiment[sentence]

	
	if num_sentences == 0:
		return 0 

	# Normalize, get avg sentiment per sentence
	pred_sentiment = float(pred_sentiment) / num_sentences

	# print(pred_sentiment)

	if pred_sentiment >= mean + 0.5 * stdev: #0.5 * abs(mean):
		return 1
	elif pred_sentiment <= mean - 0.5 * stdev: #0.5 * abs(mean):
		return -1
	else:
		return 0



def compute_DNN_accuracy(csv_file, test_data):
	sentence_to_sentiment, mean, stdev = Read_CSV(csv_file)

	# Test on only test data
	num_reviews = len(test_data)
	num_correct = 0

	sent_correct = {-1: 0, 0: 0, 1: 0}

	for data_point in test_data:
		review = data_point[0]
		actor = data_point[1]

		actor_name = actor["name"]
		actor_sentences = actor["sentences"]
		true_sentiment = actor["sentiment"]

		pred_sentiment = GetSentiment(csv_file, actor_sentences)

		if pred_sentiment == true_sentiment:
			num_correct += 1
			sent_correct[pred_sentiment] += 1

	print(sent_correct)

	return float(num_correct) / num_reviews	




if __name__ == "__main__":
	review_data, _, _ = NLPParsing.get_review_data()
	train_data, test_data = NLPParsing.split_train_and_test(review_data)

	CNN_accuracy = compute_DNN_accuracy("./data/actor_sentences_CNN.csv", test_data)

	print("CNN accuracy", CNN_accuracy)

	RNN_accuracy = compute_DNN_accuracy("./data/actor_sentences_RNN.csv", test_data)

	print("RNN accuracy", RNN_accuracy)