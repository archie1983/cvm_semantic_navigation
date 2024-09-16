from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import requests
from PIL import Image
import time

class FlorenceInference():

    def __init__(self):
        self.processor = None
        self.model = None
        self.question = None
        self.device = None
        self.torch_dtype = None
        self.load_model()

    def load_model(self):
        if (self.model == None or self.processor == None):
            #self.model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda:0")
            #model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.float16, device_map="cuda:0")
            #model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16)
            #self.processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)


    ##
    # Prepare for a room classification question.
    ##
    def initialise_for_ai2_thor_room_classification(self):
        self.question = "<CAPTION>"

        return self.question

    ##
    # Prepare for a question about the items in the room.
    ##
    def initialise_for_item_extraction(self):
        self.question = "<OD>"
        #self.question = "<DENSE_REGION_CAPTION>"

        return self.question

    def classify_room(self, image_url, expected_answer = None):
        self.initialise_for_ai2_thor_room_classification()
        return self.do_inference(image_url, expected_answer)

    ##
    # Extract items visible in a given picture
    ##
    def extract_visible_items(self, image_url, expected_answer = None):
        self.initialise_for_item_extraction()
        return self.do_inference(image_url, expected_answer)

    def do_inference(self, image_url, expected_answer = None):
        start_time = time.time()

        image = Image.open(image_url)
        inputs = self.processor(text=self.question, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=20,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=self.question, image_size=(image.width, image.height))
        print(parsed_answer)

        if expected_answer is not None:
            print("#### " + expected_answer + " @@ url: " + image_url)
        end_time = time.time()
        print("Inference time: " + str(end_time - start_time))
        return (parsed_answer, end_time - start_time)

if __name__ == "__main__":
    cvm = FlorenceInference()
    cvm.load_model()
    cvm.initialise_for_ai2_thor_room_classification()

    #cvm.classify_room('scene_pics/train_56/9.png', "OFFICE")
    cvm.classify_room('scene_pics/train_56/12.png', "OFFICE")
    cvm.extract_visible_items('scene_pics/train_56/12.png')
    #cvm.classify_room('scene_pics/train_56/14.png', "OFFICE")
    cvm.classify_room('scene_pics/train_56/16.png', "OFFICE")
    cvm.extract_visible_items('scene_pics/train_56/16.png')
    cvm.classify_room('scene_pics/train_56/60.png', "BATHROOM")
    cvm.extract_visible_items('scene_pics/train_56/60.png')
    cvm.classify_room('scene_pics/train_56/56.png', "LIVING ROOM")
    cvm.extract_visible_items('scene_pics/train_56/56.png')
