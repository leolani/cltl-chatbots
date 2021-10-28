# cltl-chatbots
Series of chatbots that demonstrate Leolani’s functionalities.

The chatbots use the CLTL EMISSOR and  KnowledgeRepresentation (aka the BRAIN) models
and follow the Leolani platform in which signals are processed and generated 
as a stream in time. The interpretation of the signals is stored in the BRAIN,
where knowledge cumulates. Reasoning over this knowledge (aka THOUGHTS), 
triggers to responses of the system to changes in the BRAIN as a result of input signal
interpretations.

![./images/signal-to-symbolic.png](./images/signal-to-symbolic.png)


The interaction with a user is recorded by EMISSOR as signals in a scenario with a timeline. 
EMISSOR can record audio, text and images. The BRAIN is a triple store that records the
interpretation and cumulated knowledge but also the perspective of the users.

![./images/interaction-to-knowledge.png](./images/interaction-to-knowledge.png)

Several Jupyter notebooks have been included that demonstrate different types of interactions.

## Getting started

Before starting install GraphDB and launch it with a sandbox repository, 
which will act as a brain. A free version of GraphDB can be donwloaded and installed from:

https://graphdb.ontotext.com

After installing GraphDB you need to launch and create a repository with the name "sandbox".
This repository will be used as the BRAIN.

## Installing

In order to install the packages you should do the following from the terminal:

* If you do not have virtualenv for Python installed install virtualenv for Python: pip install --user virtualenv
  
Please read about virtual environments if you are ot familiar with it: https://janakiev.com/blog/jupyter-virtual-envs/

* clone this github repository in a folder of your choice": git clone git@github.com:leolani/cltl-chatbots.git
* cd to clt-chatbots where the code is cloned
* create a virtual anvironment within your cloned folder: python -m venv venv
* activate the environment: source venv/bin/activate
* pip install --upgrade pip
* pip install -r requirements.txt
  
When there are no error messages you can launch jupyter to load the notebooks. 
However, you need to tell Jupyter to use the kernel for the venv where you installed all the stuff:

* python -m ipykernel install --user --name=venv

## Running the notebooks:

Start jupyter and select the kernel venv

* jupyter lab
* select kernel venv for each notebook

## References

When using this code please make reference to the following papers:

@article{santamaria2021emissor,
  title={EMISSOR: A platform for capturing multimodal interactions as Episodic Memories and Interpretations with Situated Scenario-based Ontological References},
  author={Santamar{\'\i}a, Selene B{\'a}ez and Baier, Thomas and Kim, Taewoon and Krause, Lea and Kruijt, Jaap and Vossen, Piek},
  booktitle={Processings of the MMSR workshop "Beyond Language: Multimodal Semantic Representations", IWSC2021, also available as arXiv preprint arXiv:2105.08388},
  year={2021}
}

@inproceedings{vossen2019modelling,
  title={Modelling context awareness for a situated semantic agent},
  author={Vossen, Piek and Baj{\v{c}}eti{\'c}, Lenka and Baez, Selene and Ba{\v{s}}i{\'c}, Suzana and Kraaijeveld, Bram},
  booktitle={International and Interdisciplinary Conference on Modeling and Using Context},
  pages={238--252},
  year={2019},
  organization={Springer}
}
