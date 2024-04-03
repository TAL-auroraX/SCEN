import os
import yaml
import sys
import torch
import json
from tqdm import tqdm
import numpy as np
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
from utils import parent_module,get_model,get_nested_attr,check
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# set seed 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)

class Expert_learning(nn.Module):
    def __init__(self,layer, expert_learning_path=""):
        super(Expert_learning, self).__init__()
        self.layer = layer
        self.expert_learning = nn.Linear(layer.in_features, layer.out_features, bias=False)
        if expert_learning_path!="":
            print("save ft weight")
            self.expert_learning.weight = torch.load(expert_learning_path)
        else:
            print("load raw weight")
            self.expert_learning.weight = copy.deepcopy(layer.weight)
        
    def forward(self, *args):
        res = self.expert_learning(*args)
        return res

def scen_expert(gpu_id,edit_layer_name,config):
    
    model,tokenizer = get_model(config["zsRE_edit_model"])
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    modify_layer_names = edit_layer_name

    with open(config["zsRE_edit_data"], "r") as f:
        edit_data = json.load(f)

    for n, p in model.named_parameters():
        p.requires_grad = False

    # save raw layer weight
    original_layer_list = []
    for index_train,train_item in tqdm(enumerate(edit_data[:config["seq_length"]])):
        learning_prompt = "[INST]" + train_item["Q"] + "[/INST]"
        answer_prompt = train_item["A"]+ "</s>"
        single_input_ids = tokenizer.encode(learning_prompt + answer_prompt, return_tensors='pt').to(device)
        set100 = len(tokenizer.encode(learning_prompt))
        labels = single_input_ids.tolist()[0]
        labels[0:set100] = [-100]*set100
        labels_tensor = torch.tensor([labels])
        single_labels = labels_tensor

        for id_pname,pname in enumerate(modify_layer_names):
            parent = parent_module(model, pname)
            layer_name = pname.split(".")[-1]
            if len(original_layer_list)!=len(modify_layer_names):
                # first train save raw weight
                original_layer = getattr(parent, layer_name)
                original_layer_list.append(original_layer)
                setattr(parent, layer_name,Expert_learning(original_layer,expert_learning_path="").to(device))
            else:
                setattr(parent, layer_name,Expert_learning(original_layer_list[id_pname],expert_learning_path="").to(device))

        for n,p in get_nested_attr(model, pname).named_parameters():
            # only ft layer
            if n == "expert_learning.weight":
                p.requires_grad = True
            else:
                p.requires_grad = False
        check(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()
        for i in range(config["experts_step"]):
            res = model(input_ids=single_input_ids,labels=single_labels)
            loss = res.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  

        # save weight
        expert_learning_path = config["expert_save_path"]+config["lab_tag"]+"/"+edit_layer_name[0]+"/"  
        if not os.path.exists(expert_learning_path):
            os.system(f"mkdir -p {expert_learning_path}")
        for pname in modify_layer_names:
            parent = parent_module(model, pname)
            layer_name = pname.split(".")[-1]
            original_layer = getattr(parent, layer_name)
            # each sample corresponds to an expert
            torch.save(original_layer.expert_learning.weight, expert_learning_path + "expert_learning_weight{}.pt".format(index_train+1))

def main():
    # load config
    with open("config.yml","r") as f:
        config = yaml.safe_load(f)
    
    gpu_id = config["gpus"]
    edit_layer_name = config["modify_layer_names"]
    scen_expert(gpu_id,edit_layer_name,config)

if __name__ == '__main__':
    main()