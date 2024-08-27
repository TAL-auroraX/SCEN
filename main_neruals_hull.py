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
from utils import parent_module,get_model,get_nested_attr,check,get_labels,find_start_end
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# set seed 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)

class DynamicRouter(nn.Module):
    def __init__(self, layer,router_weight_path = ""):
        super(DynamicRouter, self).__init__()
        self.layer = layer
        self.input_features = layer.in_features #layer.in_features
        self.output_features = 1
        self.router = nn.Linear(self.input_features,  self.output_features, bias=False)
        
        if router_weight_path == "":
            pass
        else:
            self.router = torch.load(router_weight_path+"router_weight.pt")

    def forward(self, *args):
        x = self.layer(*args)
        self.output =  torch.sigmoid(self.router(*args))
        return x 

def get_last_token_index(tensor):
    positions = []
    for row in tensor:
        idxs = (row == 2).nonzero(as_tuple=True)[0]
        
        if len(idxs) > 0:
            first_idx = idxs[0].item()
            
            if first_idx > 0:
                positions.append(first_idx - 1)
            else:
                positions.append(-1)
        else:
            positions.append(-1)
    return positions

def scen_index_neruals(gpu_id,edit_layer_name,config):
    
    model,tokenizer = get_model(config["hull_edit_model"])
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)

    # edit model (freeze layers) 
    modify_layer_names = edit_layer_name
    
    for n, p in model.named_parameters():
        p.requires_grad = False

    with open(config["hull_edit_data"], "r") as f:
        edit_data = json.load(f)

    for n, p in model.named_parameters():
        p.requires_grad = False

    # save raw layer weight
    original_layer_list = []
    for id_pname,pname in enumerate(modify_layer_names):
            parent = parent_module(model, pname)
            layer_name = pname.split(".")[-1]
            if len(original_layer_list)!=len(modify_layer_names):
                original_layer = getattr(parent, layer_name)
                original_layer_list.append(original_layer)
                setattr(parent, layer_name,DynamicRouter(original_layer).to(device))
            else:
                setattr(parent, layer_name,DynamicRouter(original_layer_list[id_pname]).to(device))

    for n,p in get_nested_attr(model, pname).named_parameters():
        if "router" in n: 
            p.requires_grad = True
        else:
            p.requires_grad = False


    expert_router_path = config["neruals_save_path"]+config["lab_tag"]+"/"+edit_layer_name[0]+"/"
    if not os.path.exists(expert_router_path):
        os.system(f"mkdir -p {expert_router_path}")

    with open(config["hull_edit_data"], "r") as f:
        edit_data = json.load(f)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    tokenizer.pad_token = tokenizer.eos_token
    for train_index,train_item in tqdm(enumerate(edit_data[0:config["seq_length"]])):
        all_texts = []
        learning_prompt = train_item['text']
        all_texts.append(learning_prompt)
        neg_sample_indexs = train_item['history_indexs']
        for neg_index in neg_sample_indexs:
            learning_prompt = edit_data[neg_index]["text"]
            all_texts.append(learning_prompt)

        batch_input_ids = tokenizer(all_texts, padding='longest',return_tensors='pt')
        batch_encoded_input_sequence = batch_input_ids["input_ids"].to(device)
        answers_index = get_last_token_index(batch_encoded_input_sequence)

        for _ in range(config["neruals_step"]):
            res = model(input_ids=batch_encoded_input_sequence)
            for pname in modify_layer_names:
                parent = parent_module(model, pname)
                layer_name = pname.split(".")[-1]
                original_layer = getattr(parent, layer_name)
            # positive sample
            pos_index = answers_index[0]
            act_loss = torch.exp(-(original_layer.output[0,pos_index]))
            if len(answers_index)>1:
                disable_loss = 0
                for i,it in enumerate(answers_index):
                    if i == 0:
                        pass
                    else:
                        neg_sit = answers_index[i]
                        disable_loss += torch.exp(original_layer.output[i,neg_sit])+0.7
                disable_loss = disable_loss/(len(answers_index)-1)
  
                margin_loss = 0
                for i,it in enumerate(answers_index):
                    if i == 0:
                        pass
                    else:
                        neg_sit = answers_index[i]
                        margin_loss += (torch.exp(original_layer.output[i,neg_sit])-original_layer.output[0,pos_index])+0.3
                margin_loss = margin_loss/(batch_encoded_input_sequence.shape[0]-1)
                loss = act_loss + 1.0*(disable_loss + margin_loss)
            else:
                loss = act_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
        for pname in modify_layer_names:
            parent = parent_module(model, pname)
            layer_name = pname.split(".")[-1]
            original_layer = getattr(parent, layer_name)
            torch.save(original_layer.router.weight, expert_router_path + "expert_router{}.pt".format(len(train_item['history_indexs'])+1))
            setattr(parent, layer_name,DynamicRouter(original_layer_list[id_pname]).to(device))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def main():
    # load config
    with open("config_hull.yml","r") as f:
        config = yaml.safe_load(f)
    
    gpu_id = config["gpus"]
    edit_layer_name = config["modify_layer_names"]
    scen_index_neruals(gpu_id,edit_layer_name,config)

if __name__ == '__main__':
    main()
