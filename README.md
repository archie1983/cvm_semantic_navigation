This project is a continuation of work that was published in [LLM Semantic Navigation](https://github.com/archie1983/llm_semantic_navigation).
Please check out that repository for code to recreate LLM results as described in 

```
@inproceedings{elksnis2024utility,
  title={Utility of Local Quantized Large Language Models in Semantic Navigation},
  author={Elksnis, Arturs and Xue, Ziheng and Chen, Feng and Wang, Ning},
  booktitle={2024 29th International Conference on Automation and Computing (ICAC)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

The results are stored in a folder structure: "experiment_data/pkl_\<LLM\>/" where "\<LLM\>" is one 
of the LLM models that we have evaluated. These results (pickle files) have been included in 
this project, however, they can be re-created using code in the other repository.

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

**process_extracted_scenes.py** - Processes data extracted by **extract_scene_data.py** script in the
**LLM Semantic Navigation** project and re-classifies the data using CVM (VLM) models. 
The desired VLM can be specified on line 296.

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