import csv
from pprint import pprint
import NLPParsing
import math






if __name__ == "__main__":
	sentence_to_sentiment = {}
	counter = 0

	with open("actor_sentences.csv", 'r') as csvfile:
		spamreader = csv.reader(csvfile)
		for row in spamreader:
			counter += 1
			if counter == 1:
				continue
			sentence = row[0]
			sentiment = row[2]
			sentence_to_sentiment[sentence] = sentiment

	lis_movies, lis_actor_sentences, word_identifier, actor_identifier, all_reviews, actor_and_review = NLPParsing.get_review_data()

	# Test on only test data

	num_data = len(actor_and_review)
	num_train = int(math.ceil(0.8 * num_data))
	num_test = num_data - num_train

	test_data = actor_and_review[num_train:]

	for i in range(len(test_data)):
		data_point = actor_and_review[i]

		actor_name = data_point[0]
		review = data_point[1]
		true_sentiment = data_point[2]

		# Read the sentiment of each sentence the actor is in from DNN results
		for (actor_dict, rev) in lis_actor_sentences:
			if rev == review:
				for sentence in actor_dict[actor_name]:
					print(sentence)




	# for review in lis_actor_sentences:
 #        for actor in review.keys():
 #            sentiment = 0
 #            for sen in review[actor]:
 #                sentence += sen
 #            sentence += '.\n'

	

		# pred_sentiment = 0

		# for sentence in sentence_to_sentiment.keys():
		# 	if sentence in review:
		# 		pred_sentiment += 
