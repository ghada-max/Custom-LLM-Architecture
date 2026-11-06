//download dataset-Filter
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd

REPO="ghada-max/Custom-LLM-Architecture"
HD_FILES={
  
 "phase1_train": "pretrain/phase1_train-00000-of-00001.parquet",
     "phase1_eval":  "pretrain/phase1_eval-00000-of-00001.parquet",

 "phase2_train": "pretrain/phase2_train-00000-of-00001.parquet",
     "phase2_eval":  "pretrain/phase2_eval-00000-of-00001.parquet",

     
     "ft_train": "finetune/train-00000-of-00001.parquet",
     "ft_eval":  "finetune/eval-00000-of-00001.parquet",
}

TRAIN_SPLITS={"phase1_train","phase1_train"}

def download_and_filter():
    """download and apply hard cap to train splits
    pass