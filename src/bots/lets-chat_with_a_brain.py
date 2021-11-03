import pathlib
from datetime import date
from random import getrandbits

import requests
from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.combot.backend.api.discrete import UtteranceType
from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.triple_extraction.api import Chat, UtteranceHypothesis

context_id = getrandbits(8)
place_id = getrandbits(8)
location = requests.get("https://ipinfo.io").json()

capsule = {
    "chat": 1,
    "turn": 2,
    "author": "leolani",
    "utterance": "I found them. They are under the table.",
    "utterance_type": UtteranceType.STATEMENT,
    "position": "0-25",
    "subject": {"label": "pills", "type": ["object"]},
    "predicate": {"type": "located under"},
    "object": {"label": "table", "type": ["object"]},
    "perspective": {"certainty": 1, "polarity": 1, "sentiment": 0},
    "context_id": context_id,
    "date": date(2021, 3, 12),
    "place": "Carl's room",
    "place_id": place_id,
    "country": location['country'],
    "region": location['region'],
    "city": location['city'],
    "objects": [{'type': 'chair', 'confidence': 0.56, 'id': 1},
                {'type': 'table', 'confidence': 0.87, 'id': 1},
                {'type': 'pillbox', 'confidence': 0.92, 'id': 1}],
    "people": []
}


def rephrase_triple_json_for_capsule(triple: str):
    print(triple)
    subject_type = []
    object_type = []
    predicate_type = []

    if triple['subject']['type']:
        subject_type = triple['subject']['type'][0]
        if len(subject_type.split('.')) > 1:
            subject_type = subject_type.split('.')[1]
    if triple['predicate']['type']:
        predicate_type = triple['predicate']['type'][0]
        if len(predicate_type.split('.')) > 1:
            predicate_type = predicate_type.split('.')[1]
    if triple['object']['type']:
        object_type = triple['object']['type'][0]
        if len(object_type.split('.')) > 1:
            object_type = object_type.split('.')[1]

    rephrase = {
        "subject": {'label': triple['subject']['label'], 'type': subject_type},
        "predicate": {'label': triple['predicate']['label'], 'type': predicate_type},
        "object": {'label': triple['object']['label'], 'type': object_type},
    }
    print(rephrase)
    return rephrase


log_path = pathlib.Path('./logs')

brain = LongTermMemory(address="http://localhost:7200/repositories/sandbox", log_dir=log_path, clear_all=True)

chat = Chat("Lenka")

replier = LenkaReplier()

chat.add_utterance([UtteranceHypothesis("What do I like?", 1.0)])

chat.add_utterance([UtteranceHypothesis("I have three white cats", 1.0)])
chat.last_utterance.analyze()
print(chat.last_utterance.triple)
rephrase = rephrase_triple_json_for_capsule(chat.last_utterance.triple)
print(rephrase)
# capsule mapping magic

if chat.last_utterance.type == UtteranceType.QUESTION:
    response = brain.query_brain(chat.last_utterance.triple)
    response_json = brain_response_to_json(response)
    reply = replier.reply_to_question(response_json)

if chat.last_utterance.type == UtteranceType.STATEMENT:
    response = brain.update(rephrase, reason_types=True, create_label=True)
    response_json = brain_response_to_json(response)
    reply = replier.reply_to_statement(response_json, proactive=True, persist=True)
