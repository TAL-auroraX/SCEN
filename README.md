# Scalable Model Editing via Customized Expert Networks
The offical Pytorch implementation of paper [``Scalable Model Editing via Customized Expert Networks``](http://arxiv.org/abs/2404.02699) 

This work has been accepted by COLM2024!🥳

## Model Architecture

![](img/flowchart.png)

SCEN Overall Architecture. The left half of the figure represents the editing stages. Stagel is the process for training the experts and Stage2 is the process corresponding to training the indexing neurons. The right half is the inference stage, where the corresponding experts are activated by the indexing neurons to complete the subsequent inference.

## Requirements
see requirements.txt

## Quick Start

### ZsRE
#### Step 0. Download model
Download model🤗
https://huggingface.co/AuroraX/SCEN_ZsRE_llama2_7b

Modify the "zsRE_edit_model" field in config.yml

#### Step 1. Train experts
    python main_experts.py

#### Step 2. Train indexing neurons
    python main_neruals.py

#### Step 3. Infer
    python infer.py

### Hallucination

#### Step 0. Download model
Download model🤗:https://huggingface.co/AuroraX/SCEN_Hallucination_llama2_7b_base

Modify the "hull_edit_model" field in **config_hull.yaml**


#### Step 1. Train experts
    python main_experts_hull.py

#### Step 2. Train indexing neurons
Note:Training index neurons may take a long time. We recommend splitting the data into multiple parts and then using multiple GPUs for training. For your convenience, we also provide trained index neuron weights.

If you choose to train，do:

    python main_neruals_hull.py

or get results quickly:<br>
##### step 2.1 Download neruals🤗

https://huggingface.co/AuroraX/SCEN_Hallucination_IndexingNeurons_llama2_7B_layer20

##### step 2.2 Change the path location in the code

Modify line 22 in the infer_hull.py file. <br>

<del>path = config["neruals_save_path"] + config["lab_tag"]+ "/" + config["modify_layer_names"][0] + "/"</del>

path = "The location of the index neuron you downloaded"

#### Step 3. Infer
    python infer_hull.py


## Citation
If you find this repository useful, please consider giving a star :star: and citation
```
@misc{yao2024scalable,
      title={Scalable Model Editing via Customized Expert Networks}, 
      author={Zihan Yao and Yu He and Tianyu Qi and Ming Li},
      year={2024},
      eprint={2404.02699},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

