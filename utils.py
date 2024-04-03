import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import re

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0

def get_nested_attr(obj, attr):
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj

def parent_module(model, pname):
    components = pname.split('.')
    parent = model
    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent

def check(model):
    for name, param in model.named_parameters():
        # print(name)
        if param.requires_grad:
            print(f"{name} is unfrozen")

def is_correct(model_opt, target):
    # for zsRE
    pred_ans = model_opt.split("[/INST]")[-1].split("</s>")[0].strip()
    if pred_ans == target:
        return True
    else:
        return False

def get_model(model_path):
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path) 
    return model,tokenizer 


def get_labels(input_tensor):

    result_tensor = torch.full_like(input_tensor, -100)

    for i, row in enumerate(input_tensor):
        
        last_29962_indices = (row == 29962).nonzero(as_tuple=False)
        if last_29962_indices.nelement() == 0:
            continue  
        last_29962_index = last_29962_indices[-1].item() + 1  
        
        if last_29962_index >= len(row):
            continue

        first_2_index_after_29962 = (row[last_29962_index:] == 2).nonzero(as_tuple=False)
        if first_2_index_after_29962.nelement() == 0:
            continue  
        first_2_index_after_29962 = first_2_index_after_29962[0].item() + last_29962_index

        result_tensor[i, last_29962_index:first_2_index_after_29962+1] = input_tensor[i, last_29962_index:first_2_index_after_29962+1]
    return result_tensor

def find_start_end(tensor):
    start_end_indices = []
    for row in tensor:
        indices = (row != -100).nonzero(as_tuple=False).squeeze()
        if len(indices) > 0:
            start = indices[0].item()
            end = indices[-1].item()
            start_end_indices.append((start, end))
        else:
            start_end_indices.append((-1, -1)) 
    return start_end_indices

def is_correct(model_opt, target):
    pred_ans = model_opt.split("[/INST]")[-1].split("</s>")[0].strip()
    if pred_ans == target:
        return True
    else:
        return False
