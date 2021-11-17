# cltl-chatbots

Series of chatbots that demonstrate Leolani’s functionalities.

The chatbots use the CLTL EMISSOR and KnowledgeRepresentation (aka the BRAIN) models and follow the Leolani platform in
which signals are processed and generated as a stream in time. The interpretation of the signals is stored in the BRAIN,
where knowledge cumulates. Reasoning over this knowledge (aka THOUGHTS), triggers to responses of the system to changes
in the BRAIN as a result of input signal interpretations.

![./images/signal-to-symbolic.png](./images/signal-to-symbolic.png)

The interaction with a user is recorded by EMISSOR as signals in a scenario with a timeline. EMISSOR can record audio,
text and images. The BRAIN is a triple store that records the interpretation and cumulated knowledge but also the
perspective of the users.

![./images/interaction-to-knowledge.png](./images/interaction-to-knowledge.png)

Several Jupyter notebooks have been included that demonstrate different types of interactions.

## Getting started

Before starting install GraphDB and launch it with a sandbox repository, which will act as a brain. A free version of
GraphDB can be donwloaded and installed from:

https://graphdb.ontotext.com

After installing GraphDB you need to launch and create a repository with the name `sandbox`. This repository will be
used as the BRAIN.

Furthermore, some of the application use docker repositories for sensor data processing suc as face and object detection.
For this, you need to install Docker desktop.  You can follow the instructions on this page: https://www.docker.com/products/docker-desktop

After installing docker desktop, we advise you to pull the docker images for sensor processing before you start. The images are rather big.
Use the docker pull command from the command line:

* docker pull tae898/yolov5 (15.84GB): object detection
* docker pull tae898/age-gender:v0.2 (4.96GB): face properties
* docker pull tae898/face-detection-recognition:v0.1 (2.9GB): face identitification

Once the docker images are loaded and running in your Docker desktop they are available to make calls from the notebooks and other code.


## Installing

In order to install the packages you should do the following from the terminal:

1. Clone this repo and do the following commands from the terminal:

``` python
git clone git@github.com:leolani/cltl-chatbots.git
```

1. "cd" to clt-chatbots where the code is cloned and create a virtual environment within your cloned folder for
   installing all required packages and modules. 
   
``` python
cd cltl-chatbots
```
   
If you do not have virtualenv for Python installed, install virtualenv for Python:

``` python
pip install --user virtualenv
```

Then you can create and activate your virtual environment called `venv` :

``` python
python -m venv venv
source venv/bin/activate
```

1. We need to make the virtual environment `venv` known to Jupyter notebooks. For this do the following:

``` python
pip install ipykernel
python -m ipykernel install --user --name=venv
```

1. Now we are ready to install the main packages `emissor` and the `brain` and al other dependent packages:

``` python
pip install --upgrade pip
pip install -r requirements.txt
```

1. [OPTIONAL] Some notebooks use spaCy. Download the appropriate language model for spaCy before starting within the
   `venv`:

``` python
python -m spacy download en_core_web_sm
```

When there are no error messages you can launch jupyter to load the notebooks.

## Running the notebooks:

Start jupyter and select the kernel `venv`

``` python
jupyter lab
```
Select kernel venv for each notebook

## References

When using this code please make reference to the following papers:

@article{santamaria2021emissor, title={EMISSOR: A platform for capturing multimodal interactions as Episodic Memories
and Interpretations with Situated Scenario-based Ontological References}, author={Santamar{\'\i}a, Selene B{\'a}ez and
Baier, Thomas and Kim, Taewoon and Krause, Lea and Kruijt, Jaap and Vossen, Piek}, booktitle={Processings of the MMSR
workshop "Beyond Language: Multimodal Semantic Representations", IWSC2021, also available as arXiv preprint arXiv:
2105.08388}, year={2021} }

@inproceedings{vossen2019modelling, title={Modelling context awareness for a situated semantic agent}, author={Vossen,
Piek and Baj{\v{c}}eti{\'c}, Lenka and Baez, Selene and Ba{\v{s}}i{\'c}, Suzana and Kraaijeveld, Bram},
booktitle={International and Interdisciplinary Conference on Modeling and Using Context}, pages={238--252}, year={2019},
organization={Springer} }

@inproceedings{vossen2019leolani,
  title={Leolani: A robot that communicates and learns about the shared world},
  author={Vossen, Piek and Baez, Selene and Bajcetic, Lenka and Basic, Suzana and Kraaijeveld, Bram},
  booktitle={2019 ISWC Satellite Tracks (Posters and Demonstrations, Industry, and Outrageous Ideas), ISWC 2019-Satellites},
  pages={181--184},
  year={2019},
  organization={CEUR-WS}
}
