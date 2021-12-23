import os
import sys
from random import choice
import pprint

from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.combot.backend.api.discrete import UtteranceType
from cltl.reply_generation.data.sentences import ELOQUENCE
from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.triple_extraction.api import Chat, UtteranceHypothesis
from emissor.representation.scenario import TextSignal, Scenario

src_path = os.path.abspath(os.path.join('../..'))
if src_path not in sys.path:
    sys.path.append(src_path)

import chatbots.util.capsule_util as c_util
import chatbots.util.text_util as t_util

# basic function that creates a capsule from a triple and stores it on the brain
# @parameters triple as a json string and the initilised brain as LongTermMemory
# Posting triples triggers thoughts that are converted to json and returned
# ADDITIONAL PARAMETERS  my_brain.update(capsule, reason_types=True, create_label=False)
# reason_types=True --> it will find subjects and objects in the Semantic web and add the type information
# create_label=True --> it will automatically create a rdfs:label property from the subject label

def post_a_triple(triple, my_brain: LongTermMemory):
    response_json = None
    response = None
    capsule = c_util.triple_to_capsule(triple, UtteranceType.STATEMENT)
    try:
        response = my_brain.update(capsule)
        response_json = brain_response_to_json(response)
    except:
        print('Error:', response)
    return capsule, response_json

def post_a_triple_label_and_type(triple, my_brain: LongTermMemory):
    response_json = None
    response = None
    capsule = c_util.triple_to_capsule(triple, UtteranceType.STATEMENT)
    try:
        response = my_brain.update(capsule, reason_types=True, create_label=True)
        response_json = brain_response_to_json(response)
    except:
        print('Error:', response)
    return capsule, response_json

def post_a_triple_and_verbalise_throughts(triple, replier: LenkaReplier, my_brain: LongTermMemory):
    reply = None
    response = None
    capsule = c_util.triple_to_capsule(triple, UtteranceType.STATEMENT)
    #print(capsule)
    try:
        response = my_brain.update(capsule) # ADDITIONAL PARAMTERS reason_types=True, create_label=False
        response_json = brain_response_to_json(response)
        reply = replier.reply_to_statement(response_json) # ADDITIONAL PARAMETERS proactive=True, persist=True
    except:
        print('Error:', response)
    if reply is None:
        reply = choice(ELOQUENCE)

    return reply

def post_a_query(triple,my_brain: LongTermMemory):
    response_json = None
    response = None

    capsule = c_util.triple_to_capsule(triple, UtteranceType.QUESTION)
    #print(capsule)
    try:
        response = my_brain.query_brain(capsule)
        response_json = brain_response_to_json(response)
    except:
        print('Error:', response)
        
    return response_json


def post_a_query_and_verbalise_answer(triple, replier: LenkaReplier, my_brain: LongTermMemory):
    reply = None
    response = None

    capsule = c_util.triple_to_capsule(triple, UtteranceType.QUESTION)
    #print(capsule)
    try:
        response = my_brain.query_brain(capsule)
        response_json = brain_response_to_json(response)
        reply = replier.reply_to_question(response_json)
    except:
        print('Error:', response)
        
    if reply is None:
        reply = choice(ELOQUENCE)

    return reply



######## Utility function to integrate calls to the brain with ESMISSOR scenarios


def process_text_and_reply(scenario: Scenario,
                           place_id: str,
                           location: str,
                           human_id: str,
                           textSignal: TextSignal,
                           chat: Chat,
                           replier: LenkaReplier,
                           my_brain: LongTermMemory,
                           print_details:False):
    reply = None
    capsule = None
    response = None
    response_json = None
    ### Next, we get all possible triples
    chat.add_utterance([UtteranceHypothesis(c_util.seq_to_text(textSignal.seq), 1.0)])
    chat.last_utterance.analyze()
    if print_details:
            print('Last utterance:', c_util.seq_to_text(textSignal.seq))

    if chat.last_utterance.triple is None:
        reply = choice(ELOQUENCE)

    else:
        capsule = c_util.scenario_utterance_and_triple_to_capsule(scenario,
                                                                  place_id,
                                                                  location,
                                                                  textSignal,
                                                                  human_id,
                                                                  chat.last_utterance.type,
                                                                  chat.last_utterance.perspective,
                                                                  chat.last_utterance.triple)

        if print_details:
            print('Triple:')
            pprint.pprint(chat.last_utterance.triple)
            print('Perspective:')
            pprint.pprint(chat.last_utterance.perspective)
            print('Capsule:')
            pprint.pprint(capsule)
        
        if chat.last_utterance.type == UtteranceType.QUESTION:
            capsule = c_util.lowcase_triple_json_for_query(capsule)
            try:
                response = my_brain.query_brain(capsule)
                response_json = brain_response_to_json(response)
                reply = replier.reply_to_question(response_json)
            except:
                print('Error:', response)
        if chat.last_utterance.type == UtteranceType.STATEMENT:
            try:
                response = my_brain.update(capsule, reason_types=True, create_label=True)
                response_json = brain_response_to_json(response)
                reply = replier.reply_to_statement(response_json, proactive=True, persist=True)
            except:
                print('Error:', response)

    return capsule, reply, response_json


def process_text_spacy_and_reply(scenario: Scenario,
                           place_id: str,
                           location: str,
                           human_id: str,
                           textSignal: TextSignal,
                           chat: Chat,
                           replier: LenkaReplier,
                           my_brain: LongTermMemory,
                           nlp,
                           print_details:False):
    replies = []
    capsule = None
    response = None
    response_json = None
    ### We first add the mentions for any entities to the signal
    entities = t_util.add_ner_annotation_with_spacy(textSignal, nlp)
    subject_objects = t_util.add_np_annotation_with_spacy(textSignal, nlp)
    

    if entities is None and subject_objects is None:
        reply = choice(ELOQUENCE)
    elif len(entities)>0:
        for entity in entities:
            capsule = c_util.scenario_text_mention_to_capsule(scenario,
                                                              place_id,
                                                              location,
                                                              imageSignal,
                                                              human_id,
                                                              entity,
                                                              "perceivedIn",
                                                              textSignal.id)


            if print_details:
                print('Capsule:')
                pprint.pprint(capsule)

            try:
                response = my_brain.update(capsule, reason_types=True, create_label=True)
                response_json = brain_response_to_json(response)
                reply = replier.reply_to_statement(response_json, proactive=True, persist=True)
                relies.append(reply)
            except:
                print('Error:', response)
    elif len(subject_objects)>0:
        for np in subject_objects:
            capsule = c_util.scenario_text_mention_to_capsule(scenario,
                                                              place_id,
                                                              location,
                                                              imageSignal,
                                                              human_id,
                                                              np,
                                                              "perceivedIn",
                                                              textSignal.id)


            if print_details:
                print('Capsule:')
                pprint.pprint(capsule)

            try:
                response = my_brain.update(capsule, reason_types=True, create_label=True)
                response_json = brain_response_to_json(response)
                reply = replier.reply_to_statement(response_json, proactive=True, persist=True)
                relies.append(reply)
            except:
                print('Error:', response)

    return replies, response_json

