from pprint import pprint
import NLPProject
import sys
import random


def read_lexicon():
	lexicon = open('SentiWordNet/SentiWordNet.txt')

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


	if score > 1:
		score = 1
	elif score < -1:
		score = -1
	elif score > -1 and score < 1:
		score = 0

	return score, num_words_missing, num_words


def sentiment_analysis_guessing_randomly():
	guess = random.randint(-1,1)

	return guess, 0, 0



if __name__ == "__main__":

	review_data, actor_sentences = NLPProject.get_review_data()

	lexicon = read_lexicon()

	total_num_actors = 0
	num_actors_missing = 0
	num_actors_retrieve_correct_sentiment = 0

	total_num_words = 0
	num_words_missing_from_lexicon = 0


	for i in range(len(review_data)):
		review = review_data[i]
		actor_to_sentences = actor_sentences[i]

		actor_info = review["actor_info"]

		actor_scores = {}

		for actor in actor_info.keys():
			total_num_actors += 1

			if actor not in actor_to_sentences:
				num_actors_missing += 1
				continue

			review_sentences = actor_to_sentences[actor]

			# score, num_words_missing, num_words = sentiment_analysis_using_lexicon(review_sentences, lexicon)
			score, num_words_missing, num_words = sentiment_analysis_guessing_randomly()

			num_words_missing_from_lexicon += num_words_missing
			total_num_words += num_words

			print score

			if score == 1 and actor_info[actor] == 'Positive':
				num_actors_retrieve_correct_sentiment += 1
			elif score == -1 and actor_info[actor] == 'Negative':
				num_actors_retrieve_correct_sentiment += 1
			elif score == 0 and actor_info[actor] == 'Neutral':
				num_actors_retrieve_correct_sentiment += 1


	print "num actors missing:", num_actors_missing
	print "total actors:", total_num_actors

	print "num words missing from lexicon:", num_words_missing_from_lexicon
	print "total num words:", total_num_words

	print "num actors retrieve correct sentiment:", num_actors_retrieve_correct_sentiment

	accuracy = float(num_actors_retrieve_correct_sentiment) / total_num_actors
	print "error:", 1 - accuracy
	print "accuracy:", accuracy

