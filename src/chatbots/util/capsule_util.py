from datetime import date
import json

from cltl.combot.backend.api.discrete import UtteranceType
from emissor.representation.scenario import ImageSignal, TextSignal, Scenario


def seq_to_text(seq):
    text = ""
    for c in seq:
        text += c
    return text

### OBSOLETE
def serialise_perspective(perspective:dict):
        sentiment = ""
        polarity = ""
        emotion = ""
        if perspective['sentiment']:
            sentiment = perspective['sentiment']
        if perspective['polarity']:
            polarity = perspective['polarity']
        if perspective['emotion']:
            emotion = perspective['emotion']
        rephrase = {
            "sentiment":sentiment,
            "polarity": polarity,
            "emotion": emotion
        }
        
        return rephrase

def triple_to_capsule (triple: str, utterance_type:UtteranceType):
    capsule = {"chat": "1",
               "turn": "1",
               "author": "me",
               "utterance": "",
               "utterance_type": utterance_type,
               "position": "",
               "context_id": "1",
               "date": date.today(),
               "place": "",
               "place_id": "",
               "country": "",
               "region": "",
               "city": "",
               "objects": [],
               "people": []
               }
    if triple:
        capsule.update(rephrase_triple_json_for_capsule(triple))       
    return capsule    

def scenario_utterance_to_capsule(scenario: Scenario,
                                  place_id: str,
                                  location: str,
                                  signal: TextSignal,
                                  author: str,
                                  subj: str,
                                  pred: str,
                                  obj: str):
    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": seq_to_text(signal.seq),
               "utterance_type": UtteranceType.STATEMENT,
               "position": "0-" + str(len(signal.seq)),  # TODO generate the true offset range
               "subject": {"label": subj, "type": "person"},
               "predicate": {"label": pred},
               "object": {"label": obj, "type": ""},
               "context_id": scenario.scenario.context,
               ##### standard elements
               "date": date.today(),
               "place": location['city'],
               "place_id": place_id,
               "country": location['country'],
               "region": location['region'],
               "city": location['city'],
               "objects":  [],
               "people":  []
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
    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": seq_to_text(signal.seq),
               "utterance_type": UtteranceType.STATEMENT,
               "position": "0-" + str(len(signal.seq)),  # TODO generate the true offset range
               "subject": {"label": subj, "type": "person"},
               "predicate": {"type": pred},
               "object": {"label": obj, "type": "object"},
               #"perspective": serialise_perspective(perspective),
               "context_id": scenario.scenario.context,
               ##### standard elements
               "date": date.today(),
               "place": location['city'],
               "place_id": place_id,
               "country": location['country'],
               "region": location['region'],
               "city": location['city'],
               "objects": [],
               "people": []
               }
     
    if perspective:
        capsule.update(perspective)
        
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
    
    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": seq_to_text(signal.seq),
               "utterance_type": utterance_type,
               "position": "0-" + str(len(signal.seq)),  # TODO generate the true offset range
               "context_id": scenario.scenario.context,
               ##### standard elements
               "date": date.today(),
               "place": location['city'],
               "place_id": place_id,
               "country": location['country'],
               "region": location['region'],
               "city": location['city'],
               "objects": [],
               "people": []
               }
    if triple:
        capsule.update(rephrase_triple_json_for_capsule(triple))
    if perspective:
        capsule.update(perspective)
        
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
    if triple['predicate']['type']:
        predicate_type = triple['predicate']['type'][0]
    if triple['object']['type']:
        object_type = triple['object']['type'][0]

    rephrase = {
        "subject": {'label': triple['subject']['label'], 'type': subject_type},
        "predicate": {'label': triple['predicate']['label'], 'type': predicate_type},
        "object": {'label': triple['object']['label'], 'type': object_type},
    }
    return rephrase



def lowcase_triple_json_for_query(capsule: dict): 
    if capsule['subject']['label']:
        capsule['subject']['label'] = capsule['subject']['label'].lower()
    if capsule['predicate']['label']:
        capsule['subject']['label'] = capsule['subject']['label'].lower()
    if capsule['object']['label']:
        capsule['subject']['label'] = capsule['subject']['label'].lower()
    return capsule


###### Hardcoded capsule for perceivedBy triple for an ImageSignal
def scenario_image_perceivedBy_triple_to_capsule(scenario: Scenario,
                                                 place_id: str,
                                                 location: str,
                                                 signal: ImageSignal,
                                                 author: str,
                                                 perspective: str,
                                                 triple: str):

    reference = signal.signal.id + "#" + str(signal.bounds)  # NOT ALLOWED
    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": "",
               "position": "image",
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
               "objects": [],
               "people": []
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

    capsule = {"chat": scenario.id,
               "turn": signal.id,
               "author": author,
               "utterance": "",
               "position": "image",
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
               "objects": [],
               "people": []
               }

    return capsule
