from pprint import pprint
import NLPProject
import sys
import random
import NLPParsing


def read_lexicon():
	lexicon = open('lib/SentiWordNet/SentiWordNet.txt')

	term_to_data = {}

	for line in lexicon:
		line = line.split('\t')

		if line[0][0] == '#':
			continue

		pos = line[0]
		ID = int(line[1])
		pos_score = float(line[2])
		neg_score = float(line[3])
		definition = line[5]
		definition = definition.rstrip()

		terms = line[4]
		words = []

		while terms.find('#') != -1:
			pound_spot = terms.find('#')

			word = terms[:pound_spot]
			sense_num = int(terms[pound_spot+1])

			words.append((word, sense_num))
			terms = terms[pound_spot+1:]
			if terms.find(' ') != -1:
				terms = terms[terms.find(' ')+1:]

		for (term,sense_num) in words:
			if term not in term_to_data:
				term_to_data[term] = []

			data = {}
			data["sense_num"] = sense_num	
			data["pos"] = pos
			data["ID"] = ID
			data["pos_score"] = pos_score
			data["neg_score"] = neg_score
			data["definition"] = definition

			term_to_data[term].append(data)

	return term_to_data


def sentiment_analysis_using_lexicon(sentences, lexicon):
	num_words_missing = 0
	num_words = 0

	score = 0

	for sentence in sentences:
		sentence = sentence.split()

		# TODO: if we wanted to use the lexicon most efficiently we should do pos tagging on the sentences
		# each word in the lexicon has multiple sentiments based on meaning and pos
		# I'm just grabbing the first one for now

		# TODO: remove punctuation (lots of "words" are just punctuation)

		for word in sentence:

			num_words += 1

			if word not in lexicon:
				num_words_missing += 1
				continue

			potential_word_sentiments = lexicon[word]

			for word_sentiment in potential_word_sentiments:
				if word_sentiment["sense_num"] != 1:
					continue

				pos_score = word_sentiment["pos_score"]
				neg_score = word_sentiment["neg_score"]

				score += (pos_score - neg_score)

				break

	avg_sentiment = 0.010794992 # over all words in lexicon
	neutral_sentiment = (num_words - num_words_missing) * avg_sentiment # The exact neutral expected score

	if num_words - num_words_missing == 0:
		score = 0
	else:
		score = score / (num_words - num_words_missing) # normalize score

	if score > avg_sentiment:
		score = 1
	elif score < -avg_sentiment:
		score = -1
	else:
		score = 0

	return score, num_words_missing, num_words


def sentiment_analysis_guessing_randomly():
	guess = random.randint(-1,1)

	return guess, 0, 0


def GetSentiment(entity, review):
	lexicon = read_lexicon()

	actor_sentences = NLPParsing.get_actor_sentences(entity, review)

	score, num_words_missing, num_words = sentiment_analysis_using_lexicon(actor_sentences, lexicon)
	# score, num_words_missing, num_words = sentiment_analysis_guessing_randomly()

	return score


if __name__ == "__main__":
	review_data, _, _ = NLPParsing.get_review_data()
	train_data, test_data = NLPParsing.split_train_and_test(review_data)

	lexicon = read_lexicon()

	num_actors_retrieve_correct_sentiment = 0

	total_num_words = 0
	num_words_missing_from_lexicon = 0

	num_correct = {-1: 0, 1: 0, 0: 0}

	for data_point in test_data:
		review = data_point[0]
		actor = data_point[1]

		actor_name = actor["name"]
		actor_sentences = actor["sentences"]
		actor_sentiment = actor["sentiment"]

		score, num_words_missing, num_words = sentiment_analysis_using_lexicon(actor_sentences, lexicon)
		# score, num_words_missing, num_words = sentiment_analysis_guessing_randomly()

		num_words_missing_from_lexicon += num_words_missing
		total_num_words += num_words

		if score == actor_sentiment:
			num_actors_retrieve_correct_sentiment += 1
			num_correct[score] += 1


	print("num words missing from lexicon:", num_words_missing_from_lexicon)
	print("total num words:", total_num_words)

	accuracy = float(num_actors_retrieve_correct_sentiment) / len(test_data)
	print("error:", 1 - accuracy)
	print("accuracy:", accuracy)


	pprint(num_correct)