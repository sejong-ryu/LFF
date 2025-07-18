# %%
%cd /drive2/ryusejong/LFF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json 
import time 
import re
import random
import types
import math
import numpy as np 
from tqdm.auto import tqdm
from util.utils import set_seed, read_data, save_result, get_answer_from_text, chat_huggingface, chat_huggingface_with_hidden_states, construct_conversation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


seed = 42
set_seed(seed)

# %%
# Load prm model (1)
HUGGINGFACE_TOKEN = "XXX"
#prm_path = "UW-Madison-Lee-Lab/Qwen-PRM800K"
prm_path = "UW-Madison-Lee-Lab/Llama-PRM800K"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Load prm model (2)
candidate_tokens = [12, 10]
tokenizer = AutoTokenizer.from_pretrained(prm_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    
prm = AutoModelForCausalLM.from_pretrained(
    prm_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=device
)
prm.generation_config.temperature=None
prm.generation_config.top_p=None
prm.eval()

# %%
# Load output file
output_path = "output/zeroshot_CoT/GSM8K_Llama-3-8B-Instruct_zeroshot_CoT_train_512_seed42_portion0.1.jsonl"
output_file = read_data(output_path)[29:32]

print(f"output file: {len(output_file)}")

# %%
# Evaluate with prm model
save_path = "reward/Llama-PRM800K/GSM8K_Llama-3-8B-Instruct_zeroshot_CoT_test_512_seed42_portion0.1.jsonl"
reward_data = dict()

for i, data in tqdm(enumerate(output_file), total=len(output_file)):
    # Extract question and reasoning
    question = data["Q"]["content"].strip()
    reasoning = data["A"]["content"]
    
    # Split in line
    reasoning_steps = [l.strip() for l in reasoning.split("\n") if l.strip()]
    
    print(f"num_steps: {len(reasoning_steps)}")
    print(reasoning_steps)
    
    # Ceate prefix format for each step
    prm_input = question + ' \n\n' + ' \n\n\n\n'.join(reasoning_steps) + ' \n\n\n\n'
    #prm_input = "$65 per balloon \u00d7 170 balloons = $10,950" + ' \n\n\n\n'
    
    # Get reward
    with torch.no_grad():
        #inputs = tokenizer(prm_inputs).to(prm.device)
        inputs = torch.tensor([tokenizer.encode(prm_input)]).to(device)
        
        logits = prm(inputs).logits[:,:,candidate_tokens]
        print(logits.shape)
        scores = logits.softmax(dim=-1)[:,:,1]
        print(scores.shape)
        step_scores = scores[inputs == 23535]
        step_probs  = step_scores.tolist()
        print(step_probs)
    
    
    # Save the reward
    """
    reward_data = {
        "index": data["index"],
        "question": question,
        "reasoning_steps": reasoning_steps,
        "prm_scores": scores,
        "pred_ans": data["pred_ans"],
        "answer": data["answer"]
    }
    """
    #save_result(reward_data, save_path)
    

# %%



