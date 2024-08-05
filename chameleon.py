from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
import requests
from PIL import Image
import time

class ChameleonInference():

    def __init__(self):
        self.processor = None
        self.model = None

    def load_model(self):
        self.model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda:0")
        #model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.float16, device_map="cuda:0")
        #model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16)
        self.processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

    def analyze_room(self, image_url, expected_answer):
        #image = Image.open(requests.get("https://fujisports.com/cdn/shop/products/FUJIADULTBJJBELTS_0003s_0000_fuji__bjjadultbelt__beltbjjwhite__white__1_1_1_3_1600x1600.jpg?v=1644942404", stream=True).raw)

        #image_url = "scene_pics/jiu_jitsu_belt_white_1.jpg"
        start_time = time.time()

        #prompt = "What kind of room is in this image?<image>"
        #prompt = "What kind of room is in this image?<image> Please answer with one word only."
        #prompt = "What kind of room is in this image?<image> Please answer with one word only and choose from the following cathegories: living_room, bathroom, office, kitchen, bedroom."
        prompt = "What kind of room is in this image?<image> Please provide reasoning for your answer and make the first word in your answer the correct label of the room."
        image = Image.open(image_url)
        inputs = self.processor(prompt, images=image, return_tensors="pt").to(self.model.device, torch.bfloat16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=100) 
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(out)
        print("#### " + expected_answer + " @@ url: " + image_url)
        end_time = time.time()
        print("Inference time: " + str(end_time - start_time))


cvm = ChameleonInference()
cvm.load_model()

#cvm.analyze_room('scene_pics/train_56/9.png', "OFFICE")
cvm.analyze_room('scene_pics/train_56/12.png', "OFFICE")
#cvm.analyze_room('scene_pics/train_56/14.png', "OFFICE")
cvm.analyze_room('scene_pics/train_56/16.png', "OFFICE")
cvm.analyze_room('scene_pics/train_56/60.png', "BATHROOM")
cvm.analyze_room('scene_pics/train_56/56.png', "LIVING ROOM")
