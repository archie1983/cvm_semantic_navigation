from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from time import time

def analyze_room(tokenizer, image_url, ground_truth_label):
    t0 = time()
    image = Image.open(image_url)
    enc_image = model.encode_image(image)
    #print(image_url + " :: " + model.answer_question(enc_image, "What kind of room is this? Please choose from: kitchen, office, bedroom, bathroom, living room", tokenizer) + " " + ground_truth_label)
    print(image_url + " :: " + model.answer_question(enc_image, "What kind of room is this? Please choose only from: kitchen, bedroom, bathroom, living room", tokenizer) + " " + ground_truth_label)
    print("llm predict time:", round(time()-t0, 3), "s")

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

#analyze_room(tokenizer, 'pictures_first_view/kitchen_1.png', "KITCHEN")
#analyze_room(tokenizer, 'pictures_first_view/livingroom_1.png', "LIVING ROOM")
#analyze_room(tokenizer, 'pictures_first_view/bathroom_1.png', "BATHROOM")
#analyze_room(tokenizer, 'pictures_first_view/type_br_2.png', "OFFICE")
#analyze_room(tokenizer, 'pictures_first_view/bathroom_2.png', "BATHROOM")
#analyze_room(tokenizer, 'pictures_first_view/bedroom_1.png', "BEDROOM")
#analyze_room(tokenizer, 'pictures_first_view/bedroom_2.png', "BEDROOM")
#analyze_room(tokenizer, 'pictures_first_view/kitchen_2.png', "KITCHEN")
#analyze_room(tokenizer, 'pictures_first_view/livingroom_2.png', "LIVING ROOM")

analyze_room(tokenizer, 'scene_pics/train_56/9.png', "OFFICE")
analyze_room(tokenizer, 'scene_pics/train_56/12.png', "OFFICE")
analyze_room(tokenizer, 'scene_pics/train_56/14.png', "OFFICE")
analyze_room(tokenizer, 'scene_pics/train_56/16.png', "OFFICE")

