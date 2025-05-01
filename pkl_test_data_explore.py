import glob
import pickle
from scene_description import SceneDescription
import shutil, os

def process_scene_files():
    pkl_store = "experiment_data/test_LLAMA/*.pkl"

    pkl_files = glob.glob(pkl_store) # files showing gemma scenes
    #scene_files2 = glob.glob(pkl_store2)

    for test_f in pkl_files:
        f = open(test_f,'rb')
        test_res = pickle.load(f)

        print(f)
        print(test_res)

process_scene_files()
