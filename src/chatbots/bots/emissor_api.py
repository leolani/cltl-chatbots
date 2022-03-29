# general imports for EMISSOR and the BRAIN
import emissor as em
from emissor.representation.scenario import ImageSignal
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario
import chatbots.util.driver_util as d_util

# specific imports
from random import getrandbits
from datetime import datetime
import time
import requests
import pathlib
import sys
import os


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
