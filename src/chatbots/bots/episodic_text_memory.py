# general imports for EMISSOR and the BRAIN
import emissor as em
from cltl import brain
from emissor.representation.scenario import ImageSignal

from cltl import brain
from cltl.reply_generation.data.sentences import GREETING, ASK_NAME, ELOQUENCE, TALK_TO_ME
from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.triple_extraction.api import Chat, UtteranceHypothesis
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario

# specific imports
from random import getrandbits, choice
from datetime import datetime
import time
import requests
import spacy
import pathlib
import sys
import os

#### The next utils are needed for the interaction and creating triples and capsules
import src.chatbots.util.driver_util as d_util
import src.chatbots.intentions.talk as talk
import src.chatbots.util.capsule_util as c_util
from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer



def get_subj_obj_labels_from_capsules(capsule_list):
    mentions = []
    for capsule in capsule_list:
        label = capsule['subject']['label']
        if label: mentions.append(label)
        label = capsule['object']['label']
        if label: mentions.append(label)
    mentions = list(set(mentions))
    return mentions

def add_mention_to_episodic_memory(textSignal: TextSignal, source, mention_list, my_brain, scenario_ctrl, location,
                                      place_id):
    response_list = []
    for mention in mention_list:

        ### We created a perceivedBy triple for this experience,
        ### @TODO we need to include the bouding box somehow in the object
        #print(mention)
        capsule = c_util.scenario_image_triple_to_capsule(scenario_ctrl,
                                                          textSignal,
                                                          location,
                                                          place_id,
                                                          source,
                                                          mention,
                                                          "denotedIn",
                                                          textSignal.id)

        #print(capsule)
        # Create the response from the system and store this as a new signal
        # We use the throughts to respond
        response = my_brain.update(capsule, reason_types=True, create_label=True)
        response_list.append(response)
    return response_list

def listen_and_remember(scenario_ctrl,
                        AGENT,
                        HUMAN_NAME,
                        HUMAN_ID,
                      my_brain,
                      location,
                      place_id):
    print_details = False
    replier = LenkaReplier()
    analyzer = CFGAnalyzer()
    chat = Chat(HUMAN_ID)
    #### Initial prompt by the system from which we create a TextSignal and store it
    initial_prompt = f"{choice(TALK_TO_ME)}"
    print(AGENT + ": " + initial_prompt)
    textSignal = d_util.create_text_signal(scenario_ctrl, initial_prompt)
    scenario_ctrl.append_signal(textSignal)

    utterance = ""
    #### Get input and loop
    while not (utterance.lower() == 'stop' or utterance.lower() == 'bye'):
        ###### Getting the next input signals
        utterance = input('\n')
        print(HUMAN_NAME + ": " + utterance)
        textSignal = d_util.create_text_signal(scenario_ctrl, utterance)
        scenario_ctrl.append_signal(textSignal)

        #### Process input and generate reply

        capsule_list, reply_list, response_list = talk.process_statement_and_reply(scenario_ctrl,
                                                     place_id,
                                                     location,
                                                     HUMAN_ID,
                                                     textSignal,
                                                     chat,
                                                     analyzer,
                                                     replier,
                                                     my_brain,
                                                     print_details)

        reply = ""
        for a_reply in reply_list:
            reply+= a_reply+". "
        print(AGENT + ": " + reply)
        textSignal = d_util.create_text_signal(scenario_ctrl, reply)
        scenario_ctrl.append_signal(textSignal)

        ###### Add denotedIn links for every subject and object label
        mention_list = get_subj_obj_labels_from_capsules(capsule_list)
        add_mention_to_episodic_memory(textSignal, HUMAN_ID, mention_list, my_brain, scenario_ctrl, location, place_id)

def start_a_scenario (AGENT:str, HUMAN_ID:str, HUMAN_NAME: str):
    ##### Setting the location

    place_id = getrandbits(8)
    location = None
    try:
        location = requests.get("https://ipinfo.io").json()
    except:
        print("failed to get the IP location")

    ### The name of your scenario
    scenario_id = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

    ### Specify the path to an existing data folder where your scenario is created and saved as a subfolder
    # Find the repository root dir
    parent, dir_name = (d_util.__file__, "_")
    while dir_name and dir_name != "src":
        parent, dir_name = os.path.split(parent)
    root_dir = parent
    scenario_path = os.path.abspath(os.path.join(root_dir, 'data'))

    if scenario_path not in sys.path:
        sys.path.append(scenario_path)

    if not os.path.exists(scenario_path):
        os.mkdir(scenario_path)
        print("Created a data folder for storing the scenarios", scenario_path)

    ### Specify the path to an existing folder with the embeddings of your friends
    friends_path = os.path.abspath(os.path.join(root_dir, 'friend_embeddings'))
    if friends_path not in sys.path:
        sys.path.append(friends_path)
        print("The paths with the friends:", friends_path)

    ### Define the folders where the images and rdf triples are saved
    imagefolder = scenario_path + "/" + scenario_id + "/" + "image"
    rdffolder = scenario_path + "/" + scenario_id + "/" + "rdf"

    ### Create the scenario folder, the json files and a scenarioStorage and scenario in memory
    scenarioStorage = d_util.create_scenario(scenario_path, scenario_id)
    scenario_ctrl = scenarioStorage.create_scenario(scenario_id, int(time.time() * 1e3), None, AGENT)
    return scenarioStorage,  scenario_ctrl, imagefolder, rdffolder, location, place_id


def main():
    ### Link your camera
    nlp = spacy.load("en_core_web_sm")
    # Initialise the brain in GraphDB

    ##### Setting the agents
    AGENT = "Leolani2"
    HUMAN_NAME = "Stranger"
    HUMAN_ID = "stranger1"
    scenarioStorage, scenario_ctrl, imagefolder, rdffolder, location, place_id = start_a_scenario(AGENT, HUMAN_ID, HUMAN_NAME)

    log_path = pathlib.Path(rdffolder)
    my_brain = brain.LongTermMemory(address="http://localhost:7200/repositories/sandbox",
                                    log_dir=log_path,
                                    clear_all=True)


    listen_and_remember(scenario_ctrl, AGENT, HUMAN_NAME, HUMAN_ID, my_brain, location, place_id)
    scenario_ctrl.scenario.ruler.end = int(time.time() * 1e3)
    scenarioStorage.save_scenario(scenario_ctrl)

if __name__ == '__main__':
    main()
    

