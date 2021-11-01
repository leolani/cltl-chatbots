import emissor as em
from cltl import brain
from cltl.triple_extraction.api import Chat, UtteranceHypothesis
from emissor.persistence import ScenarioStorage
from emissor.representation.annotation import AnnotationType, Token, NER
from emissor.representation.container import Index
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario
from cltl.brain.long_term_memory import LongTermMemory
from cltl.combot.backend.api.discrete import UtteranceType
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.reply_generation.lenka_replier import LenkaReplier

import sys
import os
src_path = os.path.abspath(os.path.join('..'))
if src_path not in sys.path:
    sys.path.append(src_path)

import util.driver_util as d_util
import util.capsule_util as c_util
###### NEXT IS NEEDED BECAUSE import helper_functions import brain_response_to_json failed for some reason
#import util.helper_functions as h_util

def process_text_and_think (scenario: Scenario, 
                  place_id:str, 
                  location: str, 
                  textSignal: TextSignal,
                  human_id: str,
                  my_brain:LongTermMemory):
    thoughts = ""
    chat = Chat(human_id)
    chat.add_utterance([UtteranceHypothesis(c_util.seq_to_text(textSignal.seq), 1.0)])
    chat.last_utterance.analyze()
    # No triple was extracted, so we missed three items (s, p, o)
            
    if chat.last_utterance.triple is None:
        utterance = "Any gossip?" + '\n'
    else:
        triple = c_util.rephrase_triple_json_for_capsule(chat.last_utterance.triple)
        # A triple was extracted so we compare it elementwise
        capsule = c_util.scenario_utterance_and_triple_to_capsule(scenario, 
                                                                  place_id,
                                                                  location,
                                                                  textSignal, 
                                                                  human_id,
                                                                  chat.last_utterance.perspective, 
                                                                  triple)
        print('Capsule:', capsule)
        thoughts = my_brain.update(capsule, reason_types=True)
        #print(thoughts)
    return thoughts    

def process_text_and_reply (scenario: Scenario, 
                            place_id:str, 
                            location: str, 
                            human_id: str, 
                            textSignal: TextSignal, 
                            replier: LenkaReplier, 
                            my_brain:LongTermMemory):
    reply = ""
    chat = Chat(human_id)
    chat.add_utterance([UtteranceHypothesis(c_util.seq_to_text(textSignal.seq), 1.0)])
    chat.last_utterance.analyze()
    
    if chat.last_utterance.triple is None:
        reply = "Sorry, did not get that."

    else:
        triple = c_util.rephrase_triple_json_for_capsule(chat.last_utterance.triple)
        capsule = c_util.scenario_utterance_and_triple_to_capsule(scenario, 
                                                                      place_id,
                                                                      location,
                                                                      textSignal, 
                                                                      human_id,
                                                                      chat.last_utterance.type,
                                                                      chat.last_utterance.perspective, 
                                                                      triple)

        print(capsule)

        # capsule mapping magic
        if chat.last_utterance.type == UtteranceType.QUESTION:
                response = my_brain.query_brain(capsule)
                response_json = brain_response_to_json(response)
                reply = replier.reply_to_question(response_json)

        if chat.last_utterance.type == UtteranceType.STATEMENT:
                response = my_brain.update(capsule, reason_types=True, create_label=True)
                response_json = brain_response_to_json(response)
                print(response_json)
                reply = replier.reply_to_statement(response_json, proactive=True, persist=True)
    return reply    


