from datetime import date

from cltl.combot.backend.api.discrete import UtteranceType
from emissor.representation.scenario import ImageSignal, TextSignal, Scenario


def seq_to_text(seq):
    text = ""
    for c in seq:
        text += c
    return text


##### Function to generate bogus elements for capsules. Without these, the update function fails
def generate_obl_object_json(human: str):
    json_string = {
        "objects": [{'type': 'chair', 'confidence': 0.59, 'id': 1},
                    {'type': 'table', 'confidence': 0.73, 'id': 1},
                    {'type': 'pillbox', 'confidence': 0.32, 'id': 1}],
        "people": [{'name': human, 'confidence': 0.98, 'id': 1}]
    }
    return json_string


def scenario_utterance_to_capsule(scenario: Scenario,
                                  place_id: str,
                                  location: str,
                                  signal: TextSignal,
                                  author: str,
                                  subj: str,
                                  pred: str,
                                  obj: str):
    value = generate_obl_object_json(author)
    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": seq_to_text(signal.seq),
               "utterance_type": UtteranceType.STATEMENT,
               "position": "0-" + str(len(signal.seq)),  # TODO generate the true offset range
               "subject": {"label": subj, "type": "person"},
               "predicate": {"type": pred},
               "object": {"label": obj, "type": "object"},
               "context_id": scenario.scenario.context,
               ##### standard elements
               "date": date.today(),
               "place": location['city'],
               "place_id": place_id,
               "country": location['country'],
               "region": location['region'],
               "city": location['city'],
               "objects": value['objects'],
               "people": value['people']
               }
    return capsule


def scenario_utterance_to_capsule_with_perspective(scenario: Scenario,
                                                   place_id: str,
                                                   location: str,
                                                   signal: TextSignal,
                                                   author: str,
                                                   perspective: str,
                                                   subj: str,
                                                   pred: str,
                                                   obj: str):
    value = generate_obl_object_json(author)
    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": seq_to_text(signal.seq),
               "utterance_type": UtteranceType.STATEMENT,
               "position": "0-" + str(len(signal.seq)),  # TODO generate the true offset range
               "subject": {"label": subj, "type": "person"},
               "predicate": {"type": pred},
               "object": {"label": obj, "type": "object"},
               "perspective": perspective,
               "context_id": scenario.scenario.context,
               ##### standard elements
               "date": date.today(),
               "place": location['city'],
               "place_id": place_id,
               "country": location['country'],
               "region": location['region'],
               "city": location['city'],
               "objects": value['objects'],
               "people": value['people']
               }
    return capsule


### create a capsule for a TextSignal with a triple and perspective string
def scenario_utterance_and_triple_to_capsule(scenario: Scenario,
                                             place_id: str,
                                             location: str,
                                             signal: TextSignal,
                                             author: str,
                                             utterance_type: UtteranceType,
                                             perspective: dict,
                                             triple: dict):
    value = generate_obl_object_json(author)
    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": seq_to_text(signal.seq),
               "utterance_type": utterance_type,
               "position": "0-" + str(len(signal.seq)),  # TODO generate the true offset range
               "subject": {'label': triple['subject']['label'], 'type': triple['subject']['type']},
               "predicate": {'type': triple['predicate']['label']},
               "object": {'label': triple['object']['label'], 'type': triple['object']['type']},
               "perspective": perspective,
               "context_id": scenario.scenario.context,
               ##### standard elements
               "date": date.today(),
               "place": location['city'],
               "place_id": place_id,
               "country": location['country'],
               "region": location['region'],
               "city": location['city'],
               "objects": value['objects'],
               "people": value['people']
               }

    return capsule


# Hack to make the triples compatible with the capsules
# {'subject': {'label': 'stranger', 'type': ['noun.person']},
# 'predicate': {'label': 'be', 'type': ['verb.stative']},
# 'object': {'label': 'Piek', 'type': ['noun.person']}}
def rephrase_triple_json_for_capsule(triple: dict):
    subject_type = []
    object_type = []
    predicate_type = []

    if triple['subject']['type']:
        subject_type = triple['subject']['type'][0]
        if len(subject_type.split('.')) > 1:
            subject_type.append(subject_type.split('.')[1])
    if triple['predicate']['type']:
        predicate_type = triple['predicate']['type'][0]
        if len(predicate_type.split('.')) > 1:
            predicate_type = predicate_type.split('.')[1]
    if triple['object']['type']:
        object_type = triple['object']['type'][0]
        if len(object_type.split('.')) > 1:
            object_type.append(object_type.split('.')[1])

    rephrase = {
        "subject": {'label': triple['subject']['label'], 'type': subject_type},
        "predicate": {'label': triple['predicate']['label'], 'type': predicate_type},
        "object": {'label': triple['object']['label'], 'type': object_type},
    }
    return rephrase


###### Hardcoded capsule for perceivedBy triple for an ImageSignal
def scenario_image_perceivedBy_triple_to_capsule(scenario: Scenario,
                                                 place_id: str,
                                                 location: str,
                                                 signal: ImageSignal,
                                                 author: str,
                                                 perspective: str,
                                                 triple: str):
    value = generate_obl_object_json(author)
    perspective = {"certainty": 1, "polarity": 1, "sentiment": 1}

    reference = signal.signal.id + "#" + str(signal.bounds)  # NOT ALLOWED
    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": "",
               "position": "image",
               "perspective": perspective,  # obligatory bogus perspective
               "subject": {"label": author, "type": "person"},
               "predicate": {"type": "perceivedBy"},
               "object": {"label": reference, "type": "string"},
               "context_id": scenario.scenario.context,
               ##### standard elements
               "date": date.today(),
               "place": location['city'],
               "place_id": place_id,
               "country": location['country'],
               "region": location['region'],
               "city": location['city'],
               "objects": value['objects'],
               "people": value['people']
               }

    return capsule


def scenario_image_triple_to_capsule(scenario: Scenario,
                                     place_id: str,
                                     location: str,
                                     signal: ImageSignal,
                                     author: str,
                                     subject: str,
                                     predicate: str,
                                     object: str):
    value = generate_obl_object_json(author)
    perspective = {"certainty": 1, "polarity": 1, "sentiment": 1}
    # reference = signal.signal.id+"#"+str(signal.bounds)  # not allowed

    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": "",
               "position": "image",
               "perspective": perspective,  # obligatory bogus perspective
               "subject": {"label": subject, "type": "person"},
               "predicate": {"type": predicate},
               "object": {"label": object, "type": "string"},
               "context_id": scenario.scenario.context,
               ##### standard elements
               "date": date.today(),
               "place": location['city'],
               "place_id": place_id,
               "country": location['country'],
               "region": location['region'],
               "city": location['city'],
               "objects": value['objects'],
               "people": value['people']
               }

    return capsule
