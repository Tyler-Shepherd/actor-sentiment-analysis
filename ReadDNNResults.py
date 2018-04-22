import csv
from pprint import pprint
import NLPParsing
import math


def compute_DNN_accuracy(csv_file, test_data):
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

	# Calculate standard deviation
	num_sentiments = len(sentiments)
	sum_sentiments = sum(sentiments)
	sentiment_mean = float(sum_sentiments) / num_sentiments
	stdev = sum([(a - sentiment_mean) ** 2 for a in sentiments])
	stdev = stdev / num_sentiments
	stdev = math.sqrt(stdev)
	print("stdev", stdev)

	# Test on only test data
	num_reviews = len(test_data)
	num_correct = 0

	for data_point in test_data:
		review = data_point[0]
		actor = data_point[1]

		actor_name = actor["name"]
		actor_sentences = actor["sentences"]
		true_sentiment = actor["sentiment"]

		pred_sentiment = 0

		# The DNN-Sentiment splits sentences weirdly so we need to check if every DNN-Sentiment result sentence is part of any sentence the actor is in
		for sentence in sentence_to_sentiment.keys():
			for actor_sentence in actor_sentences:
				if sentence in actor_sentence:
					pred_sentiment += sentence_to_sentiment[sentence]

		# DNN-Sentiment only returns positive/negative
		# So anything within one stdev/2 from 0 we rate "neutral"
		if pred_sentiment >= stdev and true_sentiment == 1:
			num_correct += 1
		elif pred_sentiment <= -stdev and true_sentiment == -1:
			num_correct += 1
		elif pred_sentiment > -stdev and pred_sentiment < stdev and true_sentiment == 0:
			num_correct += 1

	return float(num_correct) / num_reviews	




if __name__ == "__main__":
	review_data, _, _ = NLPParsing.get_review_data()
	train_data, test_data = NLPParsing.split_train_and_test(review_data)

	CNN_accuracy = compute_DNN_accuracy("actor_sentences_CNN.csv", test_data)

	print("CNN accuracy", CNN_accuracy)

	RNN_accuracy = compute_DNN_accuracy("actor_sentences_RNN.csv", test_data)

	print("RNN accuracy", RNN_accuracy)