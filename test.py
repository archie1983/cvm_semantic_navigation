from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from time import time

def analyze_room(tokenizer, image_url, ground_truth_label):
    t0 = time()
    image = Image.open(image_url)
    enc_image = model.encode_image(image)
    print(model.answer_question(enc_image, "What kind of room is this? Please choose from: kitchen, office, bedroom, bathroom, living room", tokenizer) + " " + ground_truth_label)
    print("llm predict time:", round(time()-t0, 3), "s")

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

analyze_room(tokenizer, 'pictures_first_view/1.png', "LIVING ROOM")
analyze_room(tokenizer, 'pictures_first_view/type_k_1.png', "KITCHEN")
analyze_room(tokenizer, 'pictures_first_view/type_lr_1.png', "LIVING ROOM")
analyze_room(tokenizer, 'pictures_first_view/type_bar_1.png', "BATHROOM")
analyze_room(tokenizer, 'pictures_first_view/room_type1.png', "LIVING ROOM")
analyze_room(tokenizer, 'pictures_first_view/type_br_2.png', "OFFICE")
analyze_room(tokenizer, 'pictures_first_view/type_bar_2.png', "BATHROOM")
analyze_room(tokenizer, 'pictures_first_view/type_br_1.png', "BEDROOM")
analyze_room(tokenizer, 'pictures_first_view/type_br_1.png', "BEDROOM")
analyze_room(tokenizer, 'pictures_first_view/type_k.png', "KITCHEN")
analyze_room(tokenizer, 'pictures_first_view/type_lr_2.png', "LIVING ROOM")
