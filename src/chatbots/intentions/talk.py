import os
import sys
from random import choice

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



# basic function that creates a capsule from a triple and stores it on the brain
# @parameters triple as a json string and the initilised brain as LongTermMemory
# Posting triples triggers thoughts that are converted to json and returned
# ADDITIONAL PARAMETERS  my_brain.update(capsule, reason_types=True, create_label=False)
# reason_types=True --> it will find subjects and objects in the Semantic web and add the type information
# create_label=True --> it will automatically create a rdfs:label property from the subject label

def post_a_triple(triple, my_brain: LongTermMemory):
    response_json = None
    capsule = c_util.triple_to_capsule(triple, UtteranceType.STATEMENT)
    response = my_brain.update(capsule)
    response_json = brain_response_to_json(response)
    return capsule, response_json

def post_a_triple_label_and_type(triple, my_brain: LongTermMemory):
    response_json = None
    capsule = c_util.triple_to_capsule(triple, UtteranceType.STATEMENT)
    response = my_brain.update(capsule, reason_types=True, create_label=True)
    response_json = brain_response_to_json(response)
    return capsule, response_json

def post_a_triple_and_verbalise_throughts(triple, replier: LenkaReplier, my_brain: LongTermMemory):
    reply = None
    capsule = c_util.triple_to_capsule(triple, UtteranceType.STATEMENT)
    print(capsule)
    response = my_brain.update(capsule) # ADDITIONAL PARAMTERS reason_types=True, create_label=False
    response_json = brain_response_to_json(response)
    reply = replier.reply_to_statement(response_json) # ADDITIONAL PARAMETERS proactive=True, persist=True
    
    if reply is None:
        reply = choice(ELOQUENCE)

    return reply



def post_a_query(triple,my_brain: LongTermMemory):
    response_json = None

    capsule = c_util.triple_to_capsule(triple, UtteranceType.QUESTION)
    print(capsule)
    response = my_brain.query_brain(capsule)
    response_json = brain_response_to_json(response)

    return response_json


def post_a_query_and_verbalise_answer(triple, replier: LenkaReplier, my_brain: LongTermMemory):
    reply = None

    capsule = c_util.triple_to_capsule(triple, UtteranceType.QUESTION)
    print(capsule)
    response = my_brain.query_brain(capsule)
    response_json = brain_response_to_json(response)
    reply = replier.reply_to_question(response_json)

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
                           my_brain: LongTermMemory):
    reply = None
    capsule = None
    
    chat.add_utterance([UtteranceHypothesis(c_util.seq_to_text(textSignal.seq), 1.0)])
    chat.last_utterance.analyze()

    
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

        if chat.last_utterance.type == UtteranceType.QUESTION:
            response = my_brain.query_brain(capsule)
            response_json = brain_response_to_json(response)
            reply = replier.reply_to_question(response_json)

        if chat.last_utterance.type == UtteranceType.STATEMENT:
            response = my_brain.update(capsule, reason_types=True, create_label=True)
            response_json = brain_response_to_json(response)
            reply = replier.reply_to_statement(response_json, proactive=True, persist=True)


    return capsule, reply

