from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
import requests
from PIL import Image
import time

class ChameleonInference():

    def __init__(self):
        self.processor = None
        self.model = None
        self.question = None
        self.load_model()

    def load_model(self):
        if (self.model == None or self.processor == None):
            self.model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda:0")
            #model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.float16, device_map="cuda:0")
            #model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16)
            self.processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

    ##
    # Prepare for a room classification question.
    ##
    def initialise_for_ai2_thor_room_classification(self):
        #self.question = "What kind of room is this? Please choose from: kitchen, office, bedroom, bathroom, living room"
        #prompt = "What kind of room is in this image?<image>"
        #prompt = "What kind of room is in this image?<image> Please answer with one word only."
        #prompt = "What kind of room is in this image?<image> Please answer with one word only and choose from the following cathegories: living_room, bathroom, office, kitchen, bedroom."
        self.question = "What kind of room is in this image?<image> Please provide reasoning for your answer and make the first word in your answer the correct label of the room."

        return self.question

    ##
    # Prepare for a question about the items in the room.
    ##
    def initialise_for_item_extraction(self):
        self.question = "Please give me a comma separated list of items that are in this picture!<image>"

        return self.question

    def classify_room(self, image_url, expected_answer = None):
        start_time = time.time()

        image = Image.open(image_url)
        #print(str(self.processor == None))
        inputs = self.processor(self.question, images=image, return_tensors="pt").to(self.model.device, torch.bfloat16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(out)
        if expected_answer is not None:
            print("#### " + expected_answer + " @@ url: " + image_url)
        end_time = time.time()
        print("Inference time: " + str(end_time - start_time))
        return (out, end_time - start_time)

    ##
    # Extract items visible in a given picture
    ##
    def extract_visible_items(self, image_url):
        self.initialise_for_item_extraction()

        image = Image.open(image_url)
        inputs = self.processor(self.question, images=image, return_tensors="pt").to(self.model.device, torch.bfloat16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(out)
        if expected_answer is not None:
            print("#### " + expected_answer + " @@ url: " + image_url)
        end_time = time.time()
        print("Inference time: " + str(end_time - start_time))
        return (out, end_time - start_time)

if __name__ == "__main__":
    cvm = ChameleonInference()
    cvm.load_model()

    #cvm.classify_room('scene_pics/train_56/9.png', "OFFICE")
    cvm.classify_room('scene_pics/train_56/12.png', "OFFICE")
    #cvm.classify_room('scene_pics/train_56/14.png', "OFFICE")
    cvm.classify_room('scene_pics/train_56/16.png', "OFFICE")
    cvm.classify_room('scene_pics/train_56/60.png', "BATHROOM")
    cvm.classify_room('scene_pics/train_56/56.png', "LIVING ROOM")
