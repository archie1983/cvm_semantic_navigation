from enum import Enum
from room_type import RoomType
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from time import time
from moondream import MoonDreamInference
from chameleon import ChameleonInference

class CVMType(Enum):
    MOONDREAM = 1 # Moondream
    CHAMELEON = 2 # Meta Chameleon

class CVMControl:
    def __init__(self, cvm_type):
        self.cvm_type = cvm_type
        self.cvm = None
        if self.cvm_type == CVMType.MOONDREAM:
            self.cvm = MoonDreamInference()
        elif self.cvm_type == CVMType.CHAMELEON:
            self.cvm = ChameleonInference()

    ##
    # Prepare for a room classification question.
    ##
    def initialise_for_ai2_thor_room_classification(self):
        return self.cvm.initialise_for_ai2_thor_room_classification()

    ##
    # Prepare for a question about the items in the room.
    ##
    def initialise_for_item_extraction(self):
        return self.cvm.initialise_for_item_extraction()

    ##
    # Constructs a question of whether this room is good to look for the given object
    ##
    def construct_room_qualification_question(self, what_to_look_for, where_to_look):
        return self.cvm.construct_room_qualification_question(what_to_look_for, where_to_look)

    ##
    # Extract items visible in a given picture
    ##
    def extract_visible_items(self, image_url):
        return self.cvm.extract_visible_items(image_url)

    ##
    # Classify a room by a given picture
    ##
    def classify_room(self, image_url):
        (full_answer, time_taken) = self.cvm.classify_room(image_url)
        #print("CVM answer: " + full_answer)

        print("cvm predict time:", round(time_taken, 3), "s")

        ret_answer = RoomType.parse_llm_response(full_answer)

        return (ret_answer, time_taken)
