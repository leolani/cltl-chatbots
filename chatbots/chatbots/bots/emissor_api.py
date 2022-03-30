# general imports for EMISSOR and the BRAIN
import os
import pathlib
import sys
import time
from datetime import datetime
# specific imports
from random import getrandbits

import emissor as em
import requests
from emissor.representation.scenario import (Annotation, ImageSignal, Mention,
                                             Modality, Scenario, TextSignal)

from ..util import driver_util as d_util


def start_a_scenario(AGENT: str, HUMAN_ID: str, HUMAN_NAME: str):
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
    os.makedirs("data", exist_ok=True)

    ### Specify the path to an existing folder with the embeddings of your friends

    ### Define the folders where the images and rdf triples are saved
    imagefolder = f"data/{scenario_id}/image"
    rdffolder = f"data/{scenario_id}/rdf"

    ### Create the scenario folder, the json files and a scenarioStorage and scenario in memory
    scenarioStorage = d_util.create_scenario("data", scenario_id)
    scenario_ctrl = scenarioStorage.create_scenario(
        scenario_id, int(time.time() * 1e3), None, AGENT
    )
    return scenarioStorage, scenario_ctrl, imagefolder, rdffolder, location, place_id
