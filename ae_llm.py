import ollama, torch, os
from enum import Enum
from room_type import RoomType
from ml_model_type import MLModelType
from transformers import (AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration,
                          MistralCommonBackend, BitsAndBytesConfig, AutoProcessor)
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 4000 only

class LLMType(Enum):
    GEMMA = 1
    MISTRAL_4b = 2
    MISTRAL_6b = 3
    LLAMA = 4
    QWEN3_5_08b = 5
    QWEN3_06b = 6
    QWEN3_06b_an_finetune = 7
    MISTRAL_MINISTRAL_3_8b = 8
    MISTRAL_MINISTRAL_3_4b = 9
    MISTRAL_MINISTRAL_3_4b_cor_tok = 10
    MINISTRAL_3_3b_instruct_fp8 = 11
    MINISTRAL_3_3b_reasoning_bf16 = 12
    MINISTRAL_3_3b_reasoning_nf4 = 13
    MINISTRAL_3_3b_instruct_nf4 = 14
    MINISTRAL_3_3b_instruct_nf4_bnb = 15

    #@classmethod
    def ollama_tag(self):
        if self == LLMType.GEMMA:
            return "gemma:7b-instruct-v1.1-q6_K"
        elif self == LLMType.MISTRAL_4b:
            return "mistral:7b-instruct-v0.2-q4_0"
        elif self == LLMType.MISTRAL_6b:
            return "mistral:7b-instruct-v0.2-q6_K"
        elif self == LLMType.LLAMA:
            return "llama3:8b-instruct-q6_K"
        else:
            return None

    #@classmethod
    def transformers_tag(self):
        if self == LLMType.QWEN3_5_08b:
            return "Qwen/Qwen3.5-0.8B"
        elif self == LLMType.QWEN3_06b:
            return "Qwen/Qwen3-0.6B"
        elif self == LLMType.QWEN3_06b_an_finetune:
            return "andresnowak/Qwen3-0.6B-instruction-finetuned"
        elif self == LLMType.MISTRAL_MINISTRAL_3_8b:
            return "mistralai/Ministral-3-3B-Reasoning-2512"
        elif self == LLMType.MISTRAL_MINISTRAL_3_4b:
            return "unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
        elif self == LLMType.MISTRAL_MINISTRAL_3_4b_cor_tok:
            return "unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
        elif self == LLMType.MINISTRAL_3_3b_instruct_fp8:
            return "mistralai/Ministral-3-3B-Instruct-2512"
        elif self == LLMType.MINISTRAL_3_3b_reasoning_bf16:
            return "mistralai/Ministral-3-3B-Reasoning-2512"
        elif self == LLMType.MINISTRAL_3_3b_reasoning_nf4:
            return "mistralai/Ministral-3-3B-Reasoning-2512"
        elif self == LLMType.MINISTRAL_3_3b_instruct_nf4:
            return "unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
        elif self == LLMType.MINISTRAL_3_3b_instruct_nf4_bnb:
            return "mistralai/Ministral-3-3B-Instruct-2512-BF16"
        else:
            return None

    @staticmethod
    def type_of_model():
        return MLModelType.LLM

class LLMControl:
    #MODEL_TO_USE = "gemma:7b-instruct-v1.1-q6_K"
    #MODEL_TO_USE = "llama3:8b-instruct-q6_K"

    def __init__(self, llm_type):
        self.llm_type = llm_type
        self.max_tokens = 512
        # if we have a transformers tag, then we need to load model and tokenizer weights now
        if self.llm_type.transformers_tag() is not None:
            model_name = self.llm_type.transformers_tag()

            device = torch.device("cuda:0")  # RTX 4000
            # load the tokenizer and the model

            if self.llm_type == LLMType.MISTRAL_MINISTRAL_3_8b:
                self.tokenizer = MistralCommonBackend.from_pretrained(model_name)
                self.model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map="auto"
                )
            elif self.llm_type == LLMType.MISTRAL_MINISTRAL_3_4b:
                self.max_tokens = 100
                self.tokenizer = MistralCommonBackend.from_pretrained("mistralai/Ministral-3-3B-Reasoning-2512")
                self.model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name, device_map="auto"
                )
            elif self.llm_type == LLMType.MISTRAL_MINISTRAL_3_4b_cor_tok:
                self.max_tokens = 100
                self.tokenizer = MistralCommonBackend.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512")
                self.model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name, device_map="auto"
                )
            elif (self.llm_type == LLMType.MINISTRAL_3_3b_instruct_fp8 or
                  self.llm_type == LLMType.MINISTRAL_3_3b_reasoning_bf16 or
                  self.llm_type == LLMType.MINISTRAL_3_3b_reasoning_nf4 or
                  self.llm_type == LLMType.MINISTRAL_3_3b_instruct_nf4_bnb):
                if self.llm_type == LLMType.MINISTRAL_3_3b_reasoning_nf4 or self.llm_type == LLMType.MINISTRAL_3_3b_instruct_nf4_bnb:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self.model = Mistral3ForConditionalGeneration.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        offload_buffers=True
                    )
                else:
                    self.model = Mistral3ForConditionalGeneration.from_pretrained(
                        model_name,
                        # quantization_config=quantization_config,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        offload_buffers=True
                    )

                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    fix_mistral_regex=True)
                self.tokenizer = None
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                ).to(device)

    def initialise(self):
        self.prompt_system = """
        You are a robot exploring an environment for the first time .
        You will be given an object to look for and should provide
        guidance of where to explore based on a series of
        observations . Observations will be given as a list of
        object clusters numbered 1 to N .

        Your job is to provide guidance about where we should explore
        next . For example if we are in a house and looking for a tv
        we should explore areas that typically have tv ’ s such as
        bedrooms and living rooms .

        You should always provide reasoning along with a number
        identifying where we should explore . If there are multiple
        right answers you should separate them with commas . Always
        include Reasoning : < your reasoning > and Answer : < your
        answer ( s ) >. If there are no suitable answers leave the
        space afters Answer : blank .
        """

        self.prompt_user = """
        I observe the following clusters of objects while exploring a
        house :
        1. sofa , tv , speaker
        2. desk , chair , computer
        3. sink , microwave , refrigerator

        Where should I search next if I am looking for a knife
        """

        self.prompt_assistant = """
        Reasoning : Knifes are typically kept in the kitchen and a sink ,
        microwave , and refrigerator are commonly found in kitchens
        . Therefore we should check the cluster that is likely to
        be a kitchen first .
        Answer : 3

        Other considerations

        1. Disregard the frequency of the objects listed on each line .
        If there are multiple of the same item in a cluster it
        will only be listed once in that cluster .
        2. You will only be given a list of common items found in the
        environment . You will not be given room labels . Use your
        best judgement when determining what room a cluster of
        objects is likely to belong to .
        """

        self.question = """
        I observe the following clusters of objects while exploring a house:
        1. couch
        2. wooden chair
        3. refrigerator

        Where should I search next if I am looking for a gas stove?

        You should always provide justification
        """

    def initialise_for_ai2_thor_room_classification(self):
        self.prompt_system = """
        You are a robot exploring a room for the first time .
        You will be given a list of objects that you can see in the room
        and should provide a guess of what kind of room it is . Objects
        will be given as a list separated by a comma .

        Your job is to make a guess of what room type these objects belong to .
        You will only have 4 room types to choose from: Living room , Kitchen ,
        Bedroom , Bathroom . For example if you are in a room and observing the
        following objects: Bath , Towel , Toiler Paper , Toothbrush then you
        should guess that you are in a Bathroom because you would typically
        find these objects in a Bathroom . Coverseley if you observe:
        Arm chair , TV , Couch then your guess should be a Living Room because
        you would typically find these objects in a Living Room.

        You should always provide reasoning along with a number
        identifying the guessed room . If there are multiple
        right answers you should separate them with commas . Always
        include Reasoning : < your reasoning > and Answer : < your
        answer ( s ) >. If there are no suitable answers leave the
        space afters Answer : blank .
        """

        self.prompt_user = """
        You observe the following objects while exploring a room:
        sink , microwave , refrigerator , waste bin

        What kind of room are you in

        1. Living room
        2. Kitchen
        3. Bedroom
        4. Bathroom
        """

        self.prompt_assistant = """
        Reasoning : Waste bin is typically kept in the kitchen and a sink ,
        microwave , and refrigerator are commonly found in kitchens
        . Therefore I would guess that I am in the kitchen .
        Answer : 2

        Other considerations

        1. Disregard the frequency of the objects listed on each line .
        If there are multiple of the same object in the list
        treat it as if mentioned only once .
        2. You will only be given a list of common items found in the
        environment . Use your best judgement when determining what room
        objects likely belong to .
        """

        self.question = """
        I observe the following objects while exploring a room:
        Candle, Plunger, Scrub Brush, Toilet

        What kind of room is this?

        1. Living room
        2. Kitchen
        3. Bedroom
        4. Bathroom

        You should always provide justification
        """

    def construct_classifier_question_multi_modal(self, query_words):
        template = """
        I observe the following objects while exploring a room:
        {0}, which is shown in the image.

        What kind of room is this?

        1. Living room
        2. Kitchen
        3. Bedroom
        4. Bathroom

        Please precede the final answer with a $ sign.
        """
        #    You should always provide justification
        # You should always provide justification and confidence estimate of your guess
        self.question = template.format(query_words)

        return self.question

    def construct_classifier_question_multi_modal_no_img(self, query_words):
        template = """
        I observe the following objects while exploring a room.
        {0}
        What kind of room is this?

        1. Living room
        2. Kitchen
        3. Bedroom
        4. Bathroom

        Please precede the final answer with a $ sign.
        """
        #    You should always provide justification
        # You should always provide justification and confidence estimate of your guess
        self.question = template.format(query_words)

        return self.question

    def construct_classifier_question_multi_modal_img_only(self):
        template = """
        I am exploring exploring a room, which is shown in the image
        What kind of room is this?

        1. Living room
        2. Kitchen
        3. Bedroom
        4. Bathroom

        Please precede the final answer with a $ sign.
        """
        #    You should always provide justification
        # You should always provide justification and confidence estimate of your guess
        self.question = template

        return self.question

    def construct_classifier_question(self, query_words):
        template = """
        I observe the following objects while exploring a room:
        {0}

        What kind of room is this?

        1. Living room
        2. Kitchen
        3. Bedroom
        4. Bathroom

        You should always provide justification
        """
#You should always provide justification and confidence estimate of your guess
        self.question = template.format(query_words)

        return self.question

    ##
    # Constructs a question of which room to look for the given object
    ##
    def construct_room_selector_question(self, object_to_find):
        template = """
        I am looking for:
        {0}

        Which room should I look in first?

        1. Living room
        2. Kitchen
        3. Bedroom
        4. Bathroom

        You should always provide justification
        """

        self.question = template.format(object_to_find)

        return self.question

    ##
    # Constructs a question of which room to look for the given object
    ##
    def construct_object_selector_question(self, what_to_look_for, where_to_look):
        template = """
        I am looking for:
        {0}

        Which object is it most likely to be near?

        {1}

        You should always provide justification
        """

        self.question = template.format(what_to_look_for, where_to_look)

        return self.question

    ##
    # Constructs a question of which room to look for the given object
    ##
    def construct_object_selector_question_ranking(self, what_to_look_for, where_to_look):
        template = """
        I am looking for:
        {0}

        Which object is it most likely to be near?

        {1}

        Please give rank for each object. You should always provide justification
        """

        self.question = template.format(what_to_look_for, where_to_look)

        return self.question

    ##
    # Ask LLM which object to look near for the given object. And return its
    # choice.
    ##
    def get_object_selector_answer(self, object_list):

        #print("obj list pre: " + str(object_list))

        object_list = [" " + obj + " " for obj in object_list]

        #print("obj list post: " + str(object_list))

        #print(" LLM :" + self.llm_type.ollama_tag())
        #print("Q CONTROL3: ", self.question)
        stream = ollama.chat(
            model = self.llm_type.ollama_tag(),
            #model='gemma:7b-instruct-q6_K',
            messages=[
                {"role": "user", "content": self.question}
            ],
            stream=True,
        )

        full_answer = ""
        ret_answer = None
        full_answer_unmodified = ""
        cur_chunk = ""
        #ret_answer = -1

        for chunk in stream:
          cur_chunk = chunk['message']['content']
          full_answer += cur_chunk
          print(cur_chunk, end='', flush=True)

        #full_answer = full_answer.replace(".", "")
        #if ("Answer:" in full_answer):
        #    ndx = full_answer.index("Answer:")
        #
        #    if (ndx >= 0 and len(full_answer) > ndx + 13):
        #        #ret_answer = full_answer[ndx + 12]
        #        nums = [int(s) for s in full_answer[ndx:(ndx + 18)].split() if s.isdigit()]
        #        ret_answer = nums[0]

        full_answer_unmodified = full_answer
        ##
        # Find the first occurence of any of the items in the list. This may not be perfect, but probably will do for now
        ##
        full_answer = full_answer.replace(".", " ")
        full_answer = full_answer.replace("*", " ")
        full_answer = full_answer.replace("#", " ")
        full_answer = full_answer.replace(",", " ")
        full_answer = full_answer.replace("!", " ")
        full_answer = full_answer.replace("?", " ")
        full_answer = full_answer.replace("\"", " ")
        full_answer = full_answer.replace("£", " ")
        full_answer = full_answer.replace("$", " ")
        full_answer = full_answer.replace("%", " ")
        full_answer = full_answer.replace("^", " ")
        full_answer = full_answer.replace("&", " ")
        full_answer = full_answer.replace("(", " ")
        full_answer = full_answer.replace(")", " ")
        full_answer = full_answer.replace("@", " ")
        full_answer = full_answer.replace("~", " ")
        full_answer = full_answer.replace("\'", " ")
        full_answer = full_answer.replace("\n", " ")

        #print(full_answer)
        nearest_index = 1000000
        for obj in object_list:
            if obj.upper() in full_answer.upper() and nearest_index > full_answer.upper().find(obj.upper()):
                ret_answer = obj.strip()
                nearest_index = full_answer.upper().find(obj.upper())

        print("NDX: ", str(nearest_index), " : ", full_answer, " : ", str(object_list), " ## ", ret_answer)
        return ret_answer, full_answer_unmodified

    def extract_obj_from_text(object_list, text):
        nearest_index = 1000000
        for obj in object_list:
            if obj.upper() in text.upper() and nearest_index > text.upper().find(obj.upper()):
                ret_answer = obj
                nearest_index = text.upper().find(obj.upper())
        return ret_answer

    def get_answer_structured_qry(self):
        #print("Q CONTROL1: ", self.question)
        stream = ollama.chat(
            model=self.llm_type.ollama_tag(),
            messages=[
                {"role": "system", "content": self.prompt_system},
                {"role": "user", "content": self.prompt_user},
                {"role": "assistant", "content": self.prompt_assistant},
                {"role": "user", "content": self.question}
            ],
            stream=True,
        )

        for chunk in stream:
          print(chunk['message']['content'], end='', flush=True)

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def get_answer(self, img_uri = None):
        #print("Q CONTROL2: ", self.question)
        if self.llm_type.transformers_tag() is not None:
            if self.llm_type == LLMType.QWEN3_06b_an_finetune:
                inputs = self.tokenizer(self.question, return_tensors="pt").to(self.model.device)
                output_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
                content = self.tokenizer.decode(output_ids[0][-self.max_tokens:], skip_special_tokens=True)
                full_answer = content[len(self.question):]
                thinking_content = ""
            elif self.llm_type == LLMType.MISTRAL_MINISTRAL_3_8b:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.question,
                            },
                        ],
                    },
                ]

                tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

                tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")

                output = self.model.generate(
                    **tokenized,
                    max_new_tokens=self.max_tokens,
                )[0]

                full_answer = self.tokenizer.decode(output[len(tokenized["input_ids"][0]):])
                #print(full_answer)
            elif self.llm_type == LLMType.MISTRAL_MINISTRAL_3_4b or self.llm_type == LLMType.MISTRAL_MINISTRAL_3_4b_cor_tok:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.question,
                            },
                        ],
                    },
                ]

                tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(self.model.device)

                tokenized["input_ids"] = tokenized["input_ids"].to(self.model.device)

                output = self.model.generate(
                    **tokenized,
                    max_new_tokens=self.max_tokens,
                )[0]

                full_answer = self.tokenizer.decode(output[len(tokenized["input_ids"][0]):])
                thinking_content = ""
            elif (self.llm_type == LLMType.MINISTRAL_3_3b_instruct_fp8 or
                  self.llm_type == LLMType.MINISTRAL_3_3b_reasoning_nf4 or
                  self.llm_type == LLMType.MINISTRAL_3_3b_reasoning_bf16 or
                  self.llm_type == LLMType.MINISTRAL_3_3b_instruct_nf4_bnb):

                if img_uri is not None:
                    image = Image.open(img_uri)  # .convert("RGB")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.question,
                                },
                                {"type": "image"},
                            ],
                        },
                    ]

                    text = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                    )

                    inputs = self.processor(
                        text=text,
                        images=[image],
                        return_tensors="pt",
                    )
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                    output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)[0]

                    full_answer = self.processor.decode(
                        output[len(inputs["input_ids"][0]):],
                        skip_special_tokens=False
                    )
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.question,
                                }
                            ],
                        },
                    ]

                    text = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                    )

                    inputs = self.processor(
                        text=text,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                    output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)[0]

                    full_answer = self.processor.decode(
                        output[len(inputs["input_ids"][0]):],
                        skip_special_tokens=False
                    )
                thinking_content = ""
            elif self.llm_type == LLMType.QWEN3_06b or self.llm_type == LLMType.QWEN3_5_08b:
                # The LLM that was chosen, lives on huggingface
                messages = [
                    {"role": "user", "content": self.question}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                # conduct text completion
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_tokens
                )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

                # parsing thinking content
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

                #print("thinking content:", thinking_content)
                #print("content:", content)
                full_answer = content
        else:
            # The LLM that was chosen lives on ollama
            stream = ollama.chat(
                model = self.llm_type.ollama_tag(),
                #model='gemma:7b-instruct-q6_K',
                messages=[
                    {"role": "user", "content": self.question}
                ],
                stream=True,
            )

            full_answer = ""
            cur_chunk = ""
            ret_answer = -1

            for chunk in stream:
                cur_chunk = chunk['message']['content']
                full_answer += cur_chunk
                print(cur_chunk, end='', flush=True)

            full_answer = full_answer.replace(".", "")
    #        if ("Answer:" in full_answer):
    #            ndx = full_answer.index("Answer:")
    #
    #            if (ndx >= 0 and len(full_answer) > ndx + 13):
    #                #ret_answer = full_answer[ndx + 12]
    #                nums = [int(s) for s in full_answer[ndx:(ndx + 18)].split() if s.isdigit()]
    #                ret_answer = nums[0]

        ret_answer = RoomType.parse_llm_response(full_answer)

        #print("NDX: " + str(ndx) + " : " + str(len(full_answer)) + " : " + full_answer[ndx + 12] + " ## " + ret_answer)
        return ret_answer, full_answer
