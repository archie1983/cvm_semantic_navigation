This project is a continuation of work that was published in [LLM Semantic Navigation](https://github.com/archie1983/llm_semantic_navigation).
Please check out that repository for code to recreate LLM results.
The results are stored in a folder structure: "experiment_data/pkl_\<LLM\>/" where "\<LLM\>" is one 
of the LLM models that we have evaluated. These results (pickle files) have been included in 
this project, but if you need to re-create them, then the code for that is in the other repository.

The project relies on [AI2-THOR](https://github.com/allenai/ai2thor), [ProcTHOR-10k](https://github.com/allenai/procthor-10k) dataset and several LLMs:
* Llama3
* Mistral
* Gemma.

Please set up a conda environment based on the **environment.yml** file, then pull the LLMs 
before running this code. The required versions are specified in **ae_llm.py** 
where more can be added, but currently the following versions are used:

* [gemma:7b-instruct-v1.1-q6_K](https://ollama.com/library/gemma:7b-instruct-v1.1-q6_K)
* [mistral:7b-instruct-v0.2-q4_0](https://ollama.com/library/mistral:7b-instruct-v0.2-q4_0)
* [mistral:7b-instruct-v0.2-q6_K](https://ollama.com/library/mistral:7b-instruct-v0.2-q6_K)
* [llama3:8b-instruct-q6_K](https://ollama.com/library/llama3:8b-instruct-q6_K)

You can pull them using Ollama. To do that, please install Ollama according to instructions at https://ollama.com/download.
Once that's done, use ollama command to pull the LLMs:
```
ollama pull gemma:7b-instruct-v1.1-q6_K
ollama pull mistral:7b-instruct-v0.2-q4_0
ollama pull mistral:7b-instruct-v0.2-q6_K
ollama pull llama3:8b-instruct-q6_K
```

In addition two VLMs are required
* Chameleon
* MoonDream2

These can be pulled automatically when you run the project, but you could also pull them in advance by running
the appropriate scripts from command line:
```
python chameleon.py
python moondream.py
```

If your machine is capable of running these models, you should see simple test run of image classification.

You must also install Thortils version from the referenced repository in the git submodule here. The vanilla Thortils will not work as I made several important changes. The best way to do that is to set up a conda environment for this purpose and install Thortils using:

```
git clone https://github.com/archie1983/cvm_semantic_navigation
cd cvm_semantic_navigation
git submodule init
git submodule update
cd thortils
pip install --no-cache-dir -e .
```

To demonstrate our approach, there are three main scripts:

**extract_scene_data.py** - for classifying a habitat from ProcTHOR-10k dataset. It will select 
next habitat with all 4 room types (**Kitchen**, **Living room**, **Bedroom**, **Bathroom**) 
from the training part of the dataset and then it will put the agent in random positions in 
that habitat and classify each random point belonging to a specific room- depending on objects 
observed around it. By default it will use SVC classifier requiring no GPU, but it can be easily
edited on line 280 to use a different classification method (LLM or CVM, i.e. VLM).

This will store classification results in Python pickle files in the **"experiment_data/data_collection"**
directory. Such generated pickle files have been stored in appropriate directories for analysis:

* pkl_GEMMA
* pkl_LLAMA
* pkl_MISTRAL_4b
* pkl_MISTRAL_6b

each corresponding to the LLM type that was used for location classification.

**process_extracted_scenes.py** - Processes data extracted by **extract_scene_data.py** script and 
re-classifies the data using CVM (VLM) models. The desired VLM can be specified on line 150.

**semantic_path_planner.py** - Defines a class that uses the points classified by **extract_scene_data.py** 
and **process_extracted_scenes.py** to generate a path to an object of interest. 
It also plots the generated path on top of a top-view habitat image. 
It can be run from Jupyter Notebook or command line. If you run this script on its own from command
line, then it will go through 100 common household items and run the selected LLM 100 times for each item
to perform target selection experiment in "train_55" habitat from ProcThor-10k dataset. The LLM can
be selected on line 570.

Finally **analyse_classification_results.ipynb** is where I analyzed room classification experiment 
results and generated box plots.
**generate_and_evaluate_datasets.ipynb** is where I generated the SVC.

**goal_selection_evaluation.ipynb** is where I analyzed goal selection experiment results.

**navigation_experiments_maj_revisions.ipynb** and **navigation_experiments.ipynb** contain a number
goal selection examples.