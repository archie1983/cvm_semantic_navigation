from enum import Enum
import glob

class ClassificationMethod(Enum):
    SVC = 1
    LLM = 2
    CVM = 3
    SVC_LLM = 4
    SVC_CVM = 5
    SVC_CVM_LLM = 6

    #@classmethod
    def svc_required(self):
        if self == ClassificationMethod.SVC or self == ClassificationMethod.SVC_LLM or self == ClassificationMethod.SVC_CVM  or self == ClassificationMethod.SVC_CVM_LLM:
            return True
        else:
            return False

    #@classmethod
    def llm_required(self):
        if self == ClassificationMethod.LLM or self == ClassificationMethod.SVC_LLM or self == ClassificationMethod.SVC_CVM_LLM:
            return True
        else:
            return False

    #@classmethod
    def cvm_required(self):
        if self == ClassificationMethod.CVM or self == ClassificationMethod.SVC_CVM  or self == ClassificationMethod.SVC_CVM_LLM:
            return True
        else:
            return False

class SceneManagement():
    def __init__(self, data_store_dir):
        self.data_store_dir = data_store_dir

    ##
    # Returns the highest index of scenes explored
    ##
    def last_index_extracted(self, pkl_files_glob = ""):
        if (pkl_files_glob == ""):
            pkl_files_glob = self.data_store_dir + "/scene_descr_train_*.pkl"

        scene_files = glob.glob(pkl_files_glob) # scene files

        highest_index = 0
        cur_index = 0

        for file_name in scene_files:
            els = file_name.split("_")
            cur_index = int(els[-1][:-4])
            if (cur_index > highest_index):
                highest_index = cur_index

        return highest_index

    ##
    # Returns the highest index of scenes processed by the DataSceneProcessor
    ##
    def last_index_processed(self, pkl_files_glob = ""):
        if (pkl_files_glob == ""):
            pkl_files_glob = self.data_store_dir + "/scene_results_train_*.pkl"

        scene_files = glob.glob(pkl_files_glob) # scene files

        highest_index = 0
        cur_index = 0

        for file_name in scene_files:
            els = file_name.split("_")
            cur_index = int(els[-1][:-4])
            if (cur_index > highest_index):
                highest_index = cur_index

        return highest_index
