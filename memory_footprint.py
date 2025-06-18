models = [('Gemma', 8.54 * 10**9, 6), # 0
          ('Mistral_4b', 7.24 * 10**9, 4), #1
          ('Mistral_6b', 7.24 * 10**9, 6), #2
          ('Llama3', 8.03 * 10**9, 6), #3
          ('Chameleon_16b', 7.04 * 10**9, 16), #4
          ('Chameleon_4b', 7.04 * 10**9, 4), #5
          ('Moondream2', 1.87 * 10**9, 16)] #6

def print_memory_reqs(model_params):
    mem_req = 0
    model_string = ""
    if isinstance(model_params, list):
        for mp in model_params:
            (model_name, model_mem_req) = print_memory_reqs(mp)
            mem_req += model_mem_req
            model_string = model_string + " " + model_name
    if isinstance(model_params, tuple):
        (model_string, p, q) = model_params
        mem_req = ((p * 4) / (32 / q)) * 1.2

    print(model_string, mem_req / 1024 / 1024 / 1024)
    return model_string, mem_req

#for m in models:
#    print_memory_reqs(m)
print_memory_reqs(models)
print_memory_reqs([models[6], models[3]]) ## Moondream + Llama
print_memory_reqs([models[4], models[3]]) ## Chameleon 16b + Llama
print_memory_reqs([models[5], models[3]]) ## Chameleon 4b + Llama
