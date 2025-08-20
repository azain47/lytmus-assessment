import os
import json
import numpy as np
from pathlib import Path

class Dataloader():
    def __init__(self, question_data_path: Path):
        if os.path.exists(question_data_path):
            with open(question_data_path,"r") as f:
                self.dataset = json.load(f)
        else:
            raise ValueError("Similar Questions Data path invalid.")
        
        print(f"========= Dataset loaded from {question_data_path}. =========")
        
    def get_dataset(self):
        return self.dataset
    
    def get_random_subset(self, size):
        return np.random.choice(self.dataset,int(size),replace=False).tolist()
           