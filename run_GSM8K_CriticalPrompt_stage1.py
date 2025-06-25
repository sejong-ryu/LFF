import openai
import json 
import time 
import re
import random
import numpy as np 
import os 
from tqdm.auto import tqdm
from util.utils import set_seed, read_data, save_result, get_answer_from_text, chat_huggingface, save_result_to_txt, save_result_to_txt2, save_sampled_indices_to_txt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


openai.api_key = "XXX"
HUGGINGFACE_TOKEN = "XXX"


def main(i, data, model, tokenizer=None):
    QAs = dict()
    QAs['index'] = int(i)
    
    question = data['question']
    answer = float(data['answer'])
    extractor = " Your final answer should be put between two ##, like ## 1 ## (if your final answer is 1), at the end of your response."
    question = question + " Explain your reasoning step-by-step." + extractor
    
    QAs['Q1'] = {'role': 'user', 'content': question}
    messages=[{'role': 'user', 'content': question}]
    
    response_1 = chat_huggingface(messages, model, tokenizer, max_new_tokens=256)

    QAs['A1'] = {'role': 'assistant', 'content':response_1}
    messages.append({'role': 'assistant', 'content':response_1})
    
    question = "Review your previous answer and find problems with your answer. Based on the problems you found, improve your answer. Please reiterate your answer." + extractor
    
    QAs['Q2'] = {'role': 'user', 'content': question}
    messages.append({'role': 'user', 'content': question})
    
    response_2 = chat_huggingface(messages, model, tokenizer, max_new_tokens=512)
    
    QAs['A2'] = {'role': 'assistant', 'content':response_2}
    messages.append({'role': 'assistant', 'content':response_2})

    QAs['answer'] = answer
    QAs['pred_ans1'] = get_answer_from_text(response_1)
    QAs['pred_ans2'] = get_answer_from_text(response_2)

    return QAs 


if __name__=='__main__':
    """
    Flag: 
        1: Run the dataset to collect responses from LLM. 
        2: Evaluate the results.
        3: Run and Evaluate.
    
    Dataset:
        GSM8K  
    """
    flag = 3
    seed = 42
    set_seed(seed)

    sample_portion = 1.0  # Set the portion of the dataset (use line 90, 91 and remove line 92)
    dataset = 'GSM8K'
    model_name = "Llama-3-8B-Instruct"
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if flag==1 or flag==3:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HUGGINGFACE_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.bfloat16,
        )
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        model.eval()
        model = model.to(device)
    
    input_dir = "dataset/"
    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #path_input = f'{input_dir}/{dataset}_test.jsonl'                       # Total test dataset
    path_input = f'{input_dir}/{dataset}_test_seed42_portion0.1.jsonl'      # 10% sampled test dataset
    path_output = f'{output_dir}/{dataset}_{model_name}_CriticalPrompt_stage1_test_256_seed42_portion0.1.jsonl'

    if flag==1 or flag==3:
        data = read_data(path_input)
        
        #sample_k = int(np.floor(sample_portion * len(data)))
        #sample_indices = np.sort(np.random.choice(np.arange(len(data)), size=sample_k, replace=False))
        #print(f"sample indices: {len(sample_indices)}")
        #save_sampled_indices_to_txt(seed, len(sample_indices), sample_indices, filename=f"sampled_indices_{seed}_{sample_portion}.txt")
        sample_indices = list(np.arange(len(data)))    # use the whole dataset
             
        print(f"data size: {len(sample_indices)}, output: {path_output}, OpenAI's key: {openai.api_key}, HuggingFace's token: {HUGGINGFACE_TOKEN}")
        for i in tqdm(sample_indices): 
            messages = main(i, data[i], model, tokenizer)
            save_result(messages, path_output)

    if flag==2 or flag==3:
        data_est = read_data(path_output)
        length = len(data_est)
        count_1 = 0 # The accuracy of zero shot CoT (standard) prompt.
        count_2 = 0 # the accuracy of Critical prompt.
        for i in range(length): 
            if data_est[i]['answer']==data_est[i]['pred_ans1']:
                count_1 += 1
            if data_est[i]['answer']==data_est[i]['pred_ans2']:
                count_2 += 1

        print(f"The accuracy of standard Prompt: {count_1/length*100}.")
        print(f"The accuracy of critical prompt: {count_2/length*100}.")
        
        path_txt = f'{output_dir}/{dataset}_{model_name}_CriticalPrompt_stage1_test_256_seed42_portion0.1.txt'
        save_result_to_txt(model_name, dataset, "CriticalPrompt stage1", count_1/length*100, count_2/length*100, path_txt)
