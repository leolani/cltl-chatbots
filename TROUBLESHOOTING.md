# Trouble shooting

We advise to install on Mac OS or Ubuntu. Installing on Windows 10 is possible but there may be some problems.
The following issues and solutions may be useful.

## Installing dependent modules

<ul>
<li> leolani/cltl-knowledgeextraction module can’t be installed via PyPi (pip). 
You need to clone the Github repository locally and adapt setup.py to specify the encoding type (encoding=”utf-8”).
<li> Make sure that Java is installed for leolani/cltl-knowledgeextraction. (Languages: Python95.6% , Java3.6% , Other0.8%)
<li> For cltl.brain make sure that python-Levenshtein and iribaker 0.2 packages are properly installed. Solution:
<ul>
<li> install Microsoft Visual C++ Redistributable for Visual Studio 2015-2022. 
<li> find and downloaded “python_Levenshtein‑0.12.2‑cp39‑cp39‑win_amd64.whl” file. 
<li> pip install python_Levenshtein-0.12.0-cp35-none-win_amd64.whl
</ul>
<li> In case of any issue with PyPI pip install, try to pip install git+git://github.com/leolani/cltl-{required package}.git@main
</ul>

## Runtime issues

<ul>
<li> Windows does not accept ":" in folder names. While running notebooks change colons (“:”) in the folder name with a convenient character because Windows doesn’t accept it. 
<li> Windows does not recognize the virtual environment "source" command. To activate venv environment go to "Scripts\bin\activate".
</ul>
