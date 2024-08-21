from room_type import RoomType
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time

class MoonDreamInference():

    def __init__(self):
        self.tokenizer = None

    ##
    # Start HuggingFace pipeline and get tokenizer for the CVM
    ##
    def get_tokenizer(self):
        if (self.tokenizer is None):
            model_id = "vikhyatk/moondream2"
            revision = "2024-05-20"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, revision=revision
            ).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, device_map="auto")

    ##
    # Prepare for a room classification question.
    ##
    def initialise_for_ai2_thor_room_classification(self):
        self.question = "What kind of room is this? Please choose from: kitchen, office, bedroom, bathroom, living room, storage" # prompt 2 - one word
        #self.question = "What kind of room is in this image? Please provide reasoning for your answer and make the first word in your answer the correct label of the room." # reasoning not provided
        #self.question = "What kind of room is in this image? Please provide reasoning for your answer." # prompt 1 - reasoning

        return self.question

    ##
    # Prepare for a question about the items in the room.
    ##
    def initialise_for_item_extraction(self):
        self.question = "Please give me a comma separated list of items that are in this picture!"

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

    ##
    # Extract items visible in a given picture
    ##
    def extract_visible_items(self, image_url):
        self.get_tokenizer()
        self.initialise_for_item_extraction()

        image = Image.open(image_url)
        enc_image = self.model.encode_image(image)
        full_answer = self.model.answer_question(enc_image, self.question, self.tokenizer)
        #print("CVM answer: " + full_answer)

        return full_answer

    ##
    # Classify a room by a given picture
    ##
    def classify_room(self, image_url, expected_answer = None):
        self.get_tokenizer()
        self.initialise_for_ai2_thor_room_classification()

        start_time = time.time()
        image = Image.open(image_url)
        enc_image = self.model.encode_image(image)
        full_answer = self.model.answer_question(enc_image, self.question, self.tokenizer)
        end_time = time.time()
        print("CVM answer: " + full_answer)

        #print("cvm predict time:", round(time()-t0, 3), "s")

        #ret_answer = RoomType.parse_llm_response(full_answer)

        return (full_answer, end_time - start_time)

if __name__ == "__main__":
    cvm = MoonDreamInference()
    cvm.get_tokenizer()

    #cvm.analyze_room('scene_pics/train_56/9.png', "OFFICE")
    cvm.analyze_room('scene_pics/train_56/12.png', "OFFICE")
    #cvm.analyze_room('scene_pics/train_56/14.png', "OFFICE")
    cvm.analyze_room('scene_pics/train_56/16.png', "OFFICE")
    cvm.analyze_room('scene_pics/train_56/60.png', "BATHROOM")
    cvm.analyze_room('scene_pics/train_56/56.png', "LIVING ROOM")
