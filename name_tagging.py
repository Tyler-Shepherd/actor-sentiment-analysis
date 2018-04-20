import requests
from pprint import pprint


# Gets all named entities of type PER in the given review text
def name_tagging(review_text):
	request = "http://blender02.cs.rpi.edu:3300/elisa_ie/entity_discovery_and_linking/en?input_format=plain%20text&output_format=KnowledgeGraph"

	r = requests.post(request, data = review_text)

	# print(r.url)
	# pprint(r.json())

	name_tagger = r.json()

	entity_id_PER = set()

	for entity in name_tagger["entity"]:
		if entity["entity_type"] == "PER":
			entity_id_PER.add(entity["entity_id"])

	named_entities = set()

	for entity in name_tagger["entity_mention"]:
		if entity["mention_id"] not in entity_id_PER:
			named_entities.add(entity["mention_head"])

	return named_entities