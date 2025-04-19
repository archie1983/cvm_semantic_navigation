from room_type import RoomType
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time, torch

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
                model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float16
            ).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, device_map="auto")

    ##
    # Prepare for a room classification question.
    ##
    def initialise_for_ai2_thor_room_classification(self):
        #self.question = "What kind of room is this? Please choose from: kitchen, office, bedroom, bathroom, living room, storage" # prompt 2 - one word
        #self.question = "What kind of room is in this image? Please provide reasoning for your answer and make the first word in your answer the correct label of the room." # reasoning not provided
        #self.question = "What kind of room is in this image? Please provide reasoning for your answer." # prompt 1 - reasoning
        self.question = "What kind of room is in this image? Please provide reasoning for your answer. You may choose one from the following categories: kitchen, bedroom, bathroom, living room." # p_cot_4lbl
        self.question = "What kind of room is in this image? Please provide reasoning for your answer. You may choose one from the following categories: kitchen, office, bedroom, bathroom, living room, storage." # p_cot_6lbl
        self.question = "What kind of room is in this image? You may choose one from the following categories: kitchen, bedroom, bathroom, living room." # p_nocot_4lbl
        self.question = "What kind of room is in this image? You may choose one from the following categories: kitchen, office, bedroom, bathroom, living room, storage." # p_nocot_6lbl
        self.question = "What kind of room is in this image?" #p_nocot_0lbl
        self.question = "What kind of room is in this image? Please provide reasoning for your answer." # p_cot_0lbl

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

    cvm.classify_room('pictures_first_view/bathroom_1.png', "BATHROOM")
    cvm.classify_room('pictures_first_view/bathroom_2.png', "BATHROOM")
    cvm.classify_room('pictures_first_view/bedroom_1.png', "BEDROOM")
    cvm.classify_room('pictures_first_view/bedroom_2.png', "BEDROOM")
    cvm.classify_room('pictures_first_view/kitchen_1.png', "KITCHEN")
    cvm.classify_room('pictures_first_view/kitchen_2.png', "KITCHEN")
    cvm.classify_room('pictures_first_view/livingroom_1.png', "LIVINGROOM")
    cvm.classify_room('pictures_first_view/livingroom_2.png', "LIVINGROOM")

