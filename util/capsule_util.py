from datetime import date
from random import choice, sample, randint, uniform
from random import getrandbits
import numpy as np

from cltl.brain import RdfBuilder, Perspective
from cltl.brain.utils.base_cases import visuals
from cltl.combot.backend.api.discrete import UtteranceType, Emotion
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario

#TEST_IMG = np.zeros((128,))
#TEST_BOUNDS = Bounds(0.0, 0.0, 0.5, 1.0)

name = 'Leolani'
places = ['Forest', 'Playground', 'Monastery', 'House', 'University', 'Hotel', 'Office']
friends = ['Piek', 'Lenka', 'Bram', 'Suzana', 'Selene', 'Lea', 'Thomas', 'Jaap', 'Tae']
unique_detections = set([item for detection in visuals for item in detection])

binary_values = [True, False]

capsule_knows = {
    "utterance": "Bram knows Lenka",
    "subject": {"label": "bram", "type": "person"},
    "predicate": {"type": "know"},
    "object": {"label": "lenka", "type": "person"},
    "perspective": {"certainty": 1, "polarity": 1, "sentiment": 0},
    "author": "suzana",
    "chat": 6,
    "turn": 4,
    "position": "0-16",
    "date": date(2019, 3, 19)
}

capsule_is_from = {
    "utterance": "Lenka is from Serbia",
    "subject": {"label": "lenka", "type": "person"},
    "predicate": {"type": "be-from"},
    "object": {"label": "serbia", "type": "location"},
    "perspective": {"certainty": 1, "polarity": 1, "sentiment": 0},
    "author": "piek",
    "chat": 1,
    "turn": 1,
    "position": "0-25",
    "date": date(2017, 10, 24)
}

capsule_is_from_2 = {
    "utterance": "Bram is from the Netherlands",
    "subject": {"label": "bram", "type": "person"},
    "predicate": {"type": "be-from"},
    "object": {"label": "netherlands", "type": "location"},
    "perspective": {"certainty": 1, "polarity": 1, "sentiment": 0},
    "author": "piek",
    "chat": 1,
    "turn": 2,
    "position": "0-25",
    "date": date(2017, 10, 24)
}

capsule_is_from_3 = {
    "utterance": "Selene is from Mexico",
    "subject": {"label": "selene", "type": "person"},
    "predicate": {"type": "be-from"},
    "object": {"label": "mexico", "type": "location"},
    "perspective": {"certainty": 1, "polarity": 1, "sentiment": 0},
    "author": "piek",
    "chat": 1,
    "turn": 3,
    "position": "0-25",
    "date": date(2017, 10, 24)
}

capsules = [capsule_is_from, capsule_is_from_2, capsule_is_from_3, capsule_knows]


def set_place(capsule, context):
    """
    Select a location randomly if not provided
    :return: place where scene takes place
    """
    if capsule.get('place', None) is not None:
        place = capsule['place']
    else:
        place = choice(places)

    context.location._label = place

    return context


def set_objects(capsule, context):
    """
    Create objects, related to the location if possible
    :param context: Context object containing information about the ongoing scene
    :param capsule: JSON
    :return: List of Object objects that are in the scene
    """
    if capsule.get('objects', None) is not None:
        objects = [Object(obj[0], obj[1], TEST_BOUNDS, TEST_IMG) for obj in capsule['objects']]
    else:
        # Office
        if context.location.label == 'Office':
            possible_objects = ['person', 'chair', 'laptop', 'bottle', 'plant']

        # Market
        elif context.location.label == 'Market':
            possible_objects = ['person', 'apple', 'banana', 'avocado', 'strawberry']

        # Playground
        elif context.location.label == 'Playground':
            possible_objects = ['person', 'teddy bear', 'cat', 'apple', 'banana']

        # Home
        elif context.location.label == 'Home':
            possible_objects = ['person', 'table', 'pills', 'chair']

        # Anywhere else
        else:
            possible_objects = unique_detections

        # Create and add objects
        num_objects = randint(0, len(possible_objects))
        objs = sample(possible_objects, num_objects)

        objects = []
        for ob in objs:
            confidence = uniform(0, 1)
            objects.append(Object(ob, confidence, TEST_BOUNDS, TEST_IMG))

    context.add_objects(objects)

    return context


def set_people(capsule, context):
    """
    Create people present in the scene
    :return: List of Face objects present in the scene
    """
    if capsule.get('people', None) is not None:
        faces = [Face(face[0], face[1], None, TEST_BOUNDS, TEST_IMG) for face in capsule['people']]
    else:
        # Add friends
        num_people = randint(0, len(friends))
        people = sample(friends, num_people)

        faces = set()
        for peep in people:
            confidence = uniform(0, 1)
            faces.add(Face(peep, confidence, None, TEST_BOUNDS, TEST_IMG))

        # Add strangers?
        if choice(binary_values):
            confidence = uniform(0, 1)
            faces.add(Face('Stranger', confidence, None, TEST_BOUNDS, TEST_IMG))

    context.add_people(faces)

    return context


def set_chat(capsule, context):
    """
    Create a Chat object, given a JSON representation and a Context object
    :param capsule: JSON
    :param context: Context object
    :return: Chat object
    """
    chat = Chat(capsule['author'], context)
    chat.id = capsule['chat']

    return chat


def set_utterance(capsule, chat):
    """
    Create an Utterance object, given a JSON representation and a Chat object
    :param capsule: JSON
    :param chat: Chat object
    :return: Utterance object
    """
    hyp = UtteranceHypothesis(capsule['utterance'], 0.99)

    utt = Utterance(chat, [hyp], False, capsule['turn'])
    utt._type = UtteranceType.STATEMENT
    utt.turn = capsule['turn']

    return utt


def set_triple(capsule, utt):
    """
    Create a Triple object given a JSON representation, and associate it to a given Utterance
    :param capsule: JSON
    :param utt: Utterance object
    :return:
    """
    builder = RdfBuilder()
    triple = builder.fill_triple(capsule['subject'], capsule['predicate'], capsule['object'])
    utt.triple = triple

    pers = set_perspective(capsule['perspective'])
    utt.perspective = pers


def set_perspective(persp):
    sentiment = persp.get('sentiment', 0.0)
    emotion = persp.get('emotion', Emotion.NEUTRAL)

    if type(emotion) != Emotion:
        emotion = Emotion[emotion.upper()]
    return Perspective(persp.get('certainty', 1), persp.get('polarity', 1), sentiment, emotion=emotion)


def transform_capsule(capsule):
    """
    Take a JSON representation and create an Utterance object
    :param capsule: JSON
    :return: Utterance object
    """

    # Fake context
    context = Context(name, friends)

    # Set people
    context = set_people(capsule, context)

    # Set objects
    context = set_objects(capsule, context)

    # Set place
    context = set_place(capsule, context)

    # Set date
    context.set_datetime(capsule['date'])

    # Set chat
    chat = set_chat(capsule, context)

    # Set utterance
    utt = set_utterance(capsule, chat)

    # Set triple
    set_triple(capsule, utt)

    return utt


def seq_to_text (seq):
    text = ""
    for c in seq:
        text+=c
    return text

def generate_obl_object_json(human:str):
    json_string ={
        "objects": [{'type': 'chair', 'confidence': 0.59, 'id': 1},
                    {'type': 'table', 'confidence': 0.73, 'id': 1},
                    {'type': 'pillbox', 'confidence': 0.32, 'id': 1}],
        "people": [{'name': human, 'confidence': 0.98, 'id': 1}]
            }
    return json_string

def scenario_utterance_to_capsule(scenario: Scenario, 
                                  place_id: str, 
                                  locatio: str, 
                                  signal: TextSignal, 
                                  author:str, 
                                  perspective:str, subj: str, 
                                  pred:str, obj:str):
    value = generate_obl_object_json(author)
    capsule = {"chat":scenario.id,
                   "turn":signal.id,
                   "author": "carl",
                    "utterance": seq_to_text(signal.seq),
                    "utterance_type": UtteranceType.STATEMENT,
                    "position": "0-"+str(len(signal.seq)),  #TODO generate the true offset range
                    "subject": {"label": subj, "type": "person"},
                    "predicate": {"type": pred},
                    "object":  {"label": obj, "type": "object"},
                    "perspective": perspective ,
                    "context_id": scenario.scenario.context,
                    "date": date.today(),
                    "place": location['city'],
                    "place_id": place_id,
                    "country": location['country'],
                    "region": location['region'],
                    "city": location['city'],
                    "objects":value['objects'],
                    "people":value['people']
                  }
    return capsule


def scenario_utterance_and_triple_to_capsule(scenario: Scenario, 
                                             place_id: str,
                                             location: str,
                                             signal: TextSignal, 
                                             author:str, 
                                             perspective:str, 
                                             triple:str):
    value = generate_obl_object_json(author)
    capsule = {"chat":scenario.id,
                   "turn":signal.id,
                   "author": "carl",
                    "utterance": seq_to_text(signal.seq),
                    "utterance_type": UtteranceType.STATEMENT,
                    "position": "0-"+str(len(signal.seq)),  #TODO generate the true offset range
                    "subject": {"label": "piek", "type": "person"},
                    "predicate": {"type": "see"},
                    "object":  {"label": "pills", "type": "object"},
                    "perspective": perspective ,
                    "context_id": scenario.scenario.context,
                    "date": date.today(),
                    "place": location['city'],
                    "place_id": place_id,
                    "country": location['country'],
                    "region": location['region'],
                    "city": location['city'],
                    "objects":value['objects'],
                    "people":value['people']
                  }
    
    return capsule

# Hack to make the triples compatible with the capsules
def rephrase_triple_json_for_capsule(triple:str):
    print(triple)
    subject_value = triple['subject']['type'][0]
    predicate_value = triple['predicate']['type'][0]
    object_value = triple['object']['type'][0]
    if len(subject_value.split('.'))>1:
        subject_value = subject_value.split('.')[1]
        
    if len(predicate_value.split('.'))>1:
        predicate_value = predicate_value.split('.')[1]
        
        
    if len(object_value.split('.'))>1:
        object_value = object_value.split('.')[1]


    rephrase = {
        "subject": {triple['subject']['label'],subject_value},
        "predicate": {triple['predicate']['label'],predicate_value},
        "object": {triple['object']['label'],object_value},


    }
    print(rephrase)
    return rephrase