import jsonpickle
import python_on_whales
import requests
import numpy as np
import pickle
import time
from PIL import Image, ImageDraw, ImageFont
from glob import glob
import logging
import os
import uuid
import platform
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity of the two vectors.

    Args
    ----
    x, y: vectors

    Returns
    -------
    similarity: a similarity score between -1 and 1, where 1 is the most
        similar.

    """

    similarity = np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

    return similarity


def start_docker_container(
    image: str, port_id: int, sleep_time=5
) -> python_on_whales.Container:
    """Start docker container given the image name and port number.

    Args
    ----
    image: docker image name
    port_id: port id
    sleep_time: warmup time

    Returns
    -------
    container: a docker container object.

    """
    container = python_on_whales.docker.run(
        image=image, detach=True, publish=[(port_id, port_id)]
    )

    logging.info(f"starting a {image} container ...")
    logging.debug(f"warming up the container ...")
    time.sleep(sleep_time)

    return container


def kill_container(container: python_on_whales.Container) -> None:
    """Kill docker container.

    Args
    ----
    container:
        a docker container object.

    """
    container.kill()
    logging.info(f"container killed.")
    logging.info(f"DONE!")


def unpickle(path: str):
    """Unpickle the pickled file, and return it.

    Args
    ----
    path: path to the pickle

    Returns
    -------
    returned: un unpickled object to be returned.

    """
    with open(path, "rb") as stream:
        returned = pickle.load(stream)

    return returned


def load_embeddings(paths: str = "./friend_embeddings/*.pkl") -> dict:
    """Load pre-defined face embeddings.

    Args
    ----
    paths: paths to the face embedding vectors.

    Returns
    -------
    embeddings_predefined: predefined embeddings

    """
    embeddings_predefined = {}
    for path in glob(paths):
        name = path.split("/")[-1].split(".pkl")[0]
        unpickled = unpickle(path)

        embeddings_predefined[unpickled["uuid"]] = {
            "embedding": unpickled["embedding"],
            "name": name,
        }

    return embeddings_predefined


def face_recognition(friends_path: str, embeddings: list, COSINE_SIMILARITY_THRESHOLD=0.65) -> list:
    """Perform face recognition based on the cosine similarity.

    Args
    ----
    embeddings: a list of embeddings
    COSINE_SIMILARITY_THRESHOLD: Currently fixed to The cosine similarity
        threshold is fixed to 0.65. Feel free to play around with this number.

    Returns
    -------
        faces_detected: list of faces (uuids and names) detected.

    """
    embeddings_predefined = load_embeddings(friends_path+"/*.pkl")

    cosine_similarities = []
    for embedding in embeddings:
        cosine_similarities_ = {
            uuid_
            + " "
            + embedding_name["name"]: cosine_similarity(
                embedding, embedding_name["embedding"]
            )
            for uuid_, embedding_name in embeddings_predefined.items()
        }
        cosine_similarities.append(cosine_similarities_)

    logging.debug(f"cosine similarities: {cosine_similarities}")

    faces_detected_ = [max(sim, key=sim.get) for sim in cosine_similarities]
    faces_detected = []
    for uuid_name, sim in zip(faces_detected_, cosine_similarities):
        uuid_, name = uuid_name.split()
        if sim[uuid_name] > COSINE_SIMILARITY_THRESHOLD:
            faces_detected.append({"uuid": uuid_, "name": name})
        else:
            logging.info("new face!")
            faces_detected.append({"uuid": str(uuid.uuid4()), "name": None})
            pass

    return faces_detected


def load_binary_image(image_path: str) -> bytes:
    """Load encoded image as a binary string and return it.

    Args
    ----
    image_path: path to the image to load.

    Returns
    -------
    binary_image: encoded binary image in bytes

    """
    logging.debug(f"{image_path} loading image ...")
    with open(image_path, "rb") as stream:
        binary_image = stream.read()
    logging.info(f"{image_path} image loaded!")

    return binary_image


def run_face_api(to_send: dict, url_face: str = "http://127.0.0.1:10002/") -> tuple:
    """Make a RESTful HTTP request to the face API server.

    Args
    ----
    to_send: dictionary to send to the server. In this function, this will be
        encoded with jsonpickle. I know this is not conventional, but encoding
        and decoding is so easy with jsonpickle somehow.
    url_face: the url of the face recognition server.

    Returns
    -------
    bboxes: (list) boudning boxes
    det_scores: (list) detection scores
    landmarks: (list) landmarks
    embeddings: (list) face embeddings
    """
    logging.debug(f"sending image to server...")
    to_send = jsonpickle.encode(to_send)
    response = requests.post(url_face, json=to_send)
    logging.info(f"got {response} from server!...")

    response = jsonpickle.decode(response.text)

    face_detection_recognition = response["face_detection_recognition"]
    logging.info(f"{len(face_detection_recognition)} faces deteced!")

    bboxes = [fdr["bbox"] for fdr in face_detection_recognition]
    det_scores = [fdr["det_score"] for fdr in face_detection_recognition]
    landmarks = [fdr["landmark"] for fdr in face_detection_recognition]

    embeddings = [fdr["normed_embedding"] for fdr in face_detection_recognition]

    return bboxes, det_scores, landmarks, embeddings



def run_age_gender_api(
    embeddings: list, url_age_gender: str = "http://127.0.0.1:10003/"
) -> tuple:
    """Make a RESTful HTTP request to the age-gender API server.

    Args
    ----
    embeddings: a list of embeddings. The number of elements in this list is
        the number of faces detected in the frame.
    url_age_gender: the url of the age-gender API server.

    Returns
    -------
    ages: (list) a list of ages
    genders: (list) a list of genders.

    """
    # -1 accounts for the batch size.
    data = np.array(embeddings).reshape(-1, 512).astype(np.float32)
    data = pickle.dumps(data)

    data = {"embeddings": data}
    data = jsonpickle.encode(data)
    logging.debug(f"sending embeddings to server ...")
    response = requests.post(url_age_gender, json=data)
    logging.info(f"got {response} from server!...")

    response = jsonpickle.decode(response.text)
    ages = response["ages"]
    genders = response["genders"]

    return ages, genders

def do_stuff_with_image(
    friends_path: str,
    image_path: str,
    url_face: str = "http://127.0.0.1:10002/",
    url_age_gender: str = "http://127.0.0.1:10003/",
) -> tuple:
    """Do stuff with image.

    Args
    ----
    image_path: path to the image in disk
    url_face: the url of the face recognition server.
    url_age_gender: the url of the age-gender API server.

    Returns
    -------
    genders
    ages
    bboxes
    faces_detected
    det_scores
    embeddings

    """
    MAXIMUM_ENTROPY = {"gender": 0.6931471805599453, "age": 4.615120516841261}

    data = {"image": load_binary_image(image_path)}

    bboxes, det_scores, landmarks, embeddings = run_face_api(data, url_face)

    faces_detected = face_recognition(friends_path, embeddings)

    ages, genders = run_age_gender_api(embeddings, url_age_gender)

    logging.debug("annotating image ...")
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    if platform.system()=='Darwin':
        font = ImageFont.truetype("/Library/fonts/Arial.ttf", 25)
    else:
        font = ImageFont.truetype("../../fonts/arial.ttf", 25)

    assert (
        len(genders)
        == len(ages)
        == len(bboxes)
        == len(faces_detected)
        == len(det_scores)
    )
    for gender, age, bbox, uuid_name, faceprob in zip(
        genders, ages, bboxes, faces_detected, det_scores
    ):
        draw.rectangle(bbox.tolist(), outline=(0, 0, 0))

        draw.text(
            (bbox[0], bbox[1]),
            f"{str(uuid_name['name']).replace('_', ' ')}, {round(age['mean'])} years old",
            fill=(255, 0, 0),
            font=font,
        )

        draw.text(
            (bbox[0], bbox[3]),
            "MALE " + str(round(gender["m"] * 100)) + str("%") + ", "
            "FEMALE " + str(round(gender["f"] * 100)) + str("%"),
            fill=(0, 255, 0),
            font=font,
        )

        image.save(image_path + ".ANNOTATED.jpg")
    logging.debug(f"image annotated and saved at {image_path + '.ANNOTATED.jpg'}")

    return genders, ages, bboxes, faces_detected, det_scores, embeddings

def add_face_annotation(imageSignal: ImageSignal,
                        container_id:str,
                        source:str,
                        mention_id: str, 
                        current_time: str,
                        bbox,
                        uri:str,
                        name:str,
                        age: str, 
                        gender:str, 
                        faceprob:str):
    
    annotations = []
    annotations.append(
                {
                    "source":source,
                    "timestamp": current_time,
                    "type": "person",
                    "value": {
                        "uri": uri,
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "faceprob": faceprob,
                    },
                }
            )

    mention_id = str(uuid.uuid4())
    segment = [
                {"bounds": bbox, "container_id": container_id, "type": "MultiIndex"}
            ]
    imageSignal.mentions.append(
                {"annotations": annotations, "id": mention_id, "segment": segment}
            )

