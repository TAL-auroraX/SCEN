# Scalable Model Editing via Customized Expert Networks
The offical Pytorch implementation of paper [``Scalable Model Editing via Customized Expert Networks``](http://arxiv.org/abs/2404.02699) 

This work has been accepted by COLM2024!🥳

## Model Architecture

![](img/flowchart.png)

SCEN Overall Architecture. The left half of the figure represents the editing stages. Stagel is the process for training the experts and Stage2 is the process corresponding to training the indexing neurons. The right half is the inference stage, where the corresponding experts are activated by the indexing neurons to complete the subsequent inference.

## Requirements


## Quick Start

### ZsRE
#### Step 1. Train experts
    python main_experts.py

#### Step 2. Train index neurons
    python main_neruals.py

#### Step 3. Infer
    python infer.py

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

