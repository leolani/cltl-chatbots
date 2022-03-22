# general imports for EMISSOR and the BRAIN
import emissor as em
from cltl import brain
from emissor.representation.scenario import ImageSignal

# specific imports
from random import getrandbits
from datetime import datetime
import time
import cv2
import requests
import spacy
import pathlib
import sys
import os

#### The next utils are needed for the interaction and creating triples and capsules
import src.chatbots.util.driver_util as d_util
import src.chatbots.util.capsule_util as c_util
import src.chatbots.util.face_util as f_util

def get_next_image(camera, imagefolder):
    what_is_seen = None

    success, frame = camera.read()
    if success:
        current_time = int(time.time() * 1e3)
        imagepath = f"{imagefolder}/{current_time}.png"
        image_bbox = (0, 0, frame.shape[1], frame.shape[0])
        cv2.imwrite(imagepath, frame)
        print(imagepath)

        what_is_seen = f_util.detect_objects(imagepath)

    return what_is_seen, current_time, image_bbox


def create_imageSignal_and_annotations_in_emissor (results, image_time, image_bbox, scenario_ctrl):

    #### We create an imageSignal
    imageSignal = d_util.create_image_signal(scenario_ctrl, f"{image_time}.png", image_bbox, image_time)
    scenario_ctrl.append_signal(imageSignal)
    what_is_seen = []
    ## The next for loop creates a capsule for each object detected in the image and posts a perceivedIn property for the object in the signal
    ## The "front_camera" is the source of the signal
    for result in results:
        current_time = int(time.time() * 1e3)

        bbox = [int(num) for num in result['yolo_bbox']]
        object_type = result['label_string']
        object_prob = result['det_score']
        what_is_seen.append(object_type)
        mention = f_util.create_object_mention(imageSignal, "front_camera", current_time, bbox, object_type,
                                               object_prob)
        imageSignal.mentions.append(mention)

    return what_is_seen, imageSignal

def add_perception_to_episodic_memory (imageSignal: ImageSignal, object_list, my_brain,  scenario_ctrl, location, place_id):
    response_list = []
    for object in object_list:
        ### We created a perceivedBy triple for this experience,
        ### @TODO we need to include the bouding box somehow in the object
        print(object)
        capsule = c_util.scenario_image_triple_to_capsule(scenario_ctrl,
                                                          imageSignal,
                                                          location,
                                                          place_id,
                                                          "front_camera",
                                                          object,
                                                          "perceivedIn",
                                                          imageSignal.id)
        
        print(capsule)
        # Create the response from the system and store this as a new signal
        # We use the throughts to respond
        response = my_brain.update(capsule, reason_types=True, create_label=True)
        response_list.append(response)
    return response_list


def watch_and_remmeber(scenario_ctrl,
          camera,
          imagefolder,
          my_brain,
          location,
          place_id):

    t1 = datetime.now()
    while (datetime.now()-t1).seconds <= 60:
        ###### Getting the next input signals
        what_did_i_see, current_time, image_bbox = get_next_image(camera, imagefolder)
        object_list, imageSignal = create_imageSignal_and_annotations_in_emissor(what_did_i_see, current_time, image_bbox, scenario_ctrl)
        response = add_perception_to_episodic_memory(imageSignal, object_list, my_brain, scenario_ctrl, location, place_id)
        print(response)
        reply = "\nI saw: "
        if len(object_list) > 1:
            for index, object in enumerate(object_list):
                if index == len(object_list) - 1:
                    reply += " and"
                reply += " a " + object
        elif len(object_list) == 1:
            reply += " a " + object_list[0]
        else:
            reply = "\nI cannot see! Something wrong with my camera."

        print(reply + "\n")


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
    camera = cv2.VideoCapture(0)
    nlp = spacy.load("en_core_web_sm")
    # Initialise the brain in GraphDB
    log_path = pathlib.Path('./logs')
    my_brain = brain.LongTermMemory(address="http://localhost:7200/repositories/sandbox",
                                    log_dir=log_path,
                                    clear_all=True)
    ##### Setting the agents
    AGENT = "Leolani2"
    HUMAN_NAME = "Stranger"
    HUMAN_ID = "stranger"
    scenarioStorage, scenario_ctrl, imagefolder, rdffolder, location, place_id = start_a_scenario(AGENT, HUMAN_ID, HUMAN_NAME)

    watch_and_remmeber(scenario_ctrl, camera, imagefolder,  my_brain, location, place_id)
    scenario_ctrl.scenario.ruler.end = int(time.time() * 1e3)
    scenarioStorage.save_scenario(scenario_ctrl)
    camera.release()

if __name__ == '__main__':
    main()
    

