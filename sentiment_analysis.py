from pprint import pprint

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

	


pprint(term_to_data)