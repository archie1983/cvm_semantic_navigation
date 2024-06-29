from enum import Enum
from room_type import RoomType
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from time import time

class CVMType(Enum):
    MOONDREAM = 1

    #@classmethod
    def model_meta(self):
        if self == CVMType.MOONDREAM:
            return ("vikhyatk/moondream2", "2024-05-20")

class CVMControl:
    def __init__(self, cvm_type):
        self.cvm_type = cvm_type
        self.get_tokenizer()

    ##
    # Start HuggingFace pipeline and get tokenizer for the CVM
    ##
    def get_tokenizer(self):
        model_id = self.cvm_type.model_meta()[0]
        revision = self.cvm_type.model_meta()[1]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def initialise_for_ai2_thor_room_classification(self):
        self.question = "What kind of room is this? Please choose from: kitchen, office, bedroom, bathroom, living room"

        return self.question

    ##
    # Constructs a question of whether this room is good to look for the given object
    ##
    def construct_room_qualification_question(self, what_to_look_for, where_to_look):
        template = """
        Is this room a good candidate to look for
        {0}
        """

        self.question = template.format(what_to_look_for)

        return self.question

    def classify_room(self, image_url):
        t0 = time()
        image = Image.open(image_url)
        enc_image = self.model.encode_image(image)
        full_answer = self.model.answer_question(enc_image, self.question, self.tokenizer)
        print("Full answer: " + full_answer)

        print("cvm predict time:", round(time()-t0, 3), "s")

        ret_answer = RoomType.parse_llm_response(full_answer)

        return ret_answer
