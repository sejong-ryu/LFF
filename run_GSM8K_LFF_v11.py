import openai
import json 
import time 
import re
import random
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" 
from tqdm.auto import tqdm
from util.utils import set_seed, read_data, save_result, get_answer_from_text, chat_huggingface, save_result_to_txt, save_result_to_txt2, get_answer_response_from_text
from util.memory_utils import retrieve_revision_advice, retrieve_error_pattern, retrieve_error_pattern2
from util.prm_utils import find_smaller_than_T
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


openai.api_key = "xxx"
HUGGINGFACE_TOKEN = "xxx"

revision_list = []

def main(i, data, model, tokenizer=None, prm=None, prm_tokenizer=None, threshold=0.6):
    QAs = dict()
    QAs['index'] = int(i)

    ### Zero-shot CoT
    question = data['question']
    
    answer = float(data['answer'])
    extractor = " Your final answer should be put between two ##, like ## 1 ## (if your final answer is 1), at the end of your response."
    question = question + " Explain your reasoning step-by-step." + extractor
    
    QAs['Q1'] = {'role': 'user', 'content': question}
    messages = [{'role': 'user', 'content': question}]
    
    response_1 = chat_huggingface(messages, model, tokenizer, max_new_tokens=512)
    
    QAs['answer'] = answer
    #QAs['pred_ans'], cut_response_1 = get_answer_from_text(response_1)
    QAs['pred_ans'], cut_response_1 = get_answer_response_from_text(response_1)
    
    #QAs['A1'] = {'role': 'assistant', 'content':response_1}
    QAs['A1'] = {'role': 'assistant', 'content':cut_response_1}
    messages.append({'role': 'assistant', 'content':cut_response_1})
    
    ### PRM
    question = QAs['Q1']['content'].strip()
    reasoning = QAs['A1']['content']   
    
    reasoning_steps = [l.strip() for l in reasoning.split("\n") if l.strip()]
    question_input_text = question + ' \n\n' + ' \n\n\n\n'.join(reasoning_steps) + ' \n\n\n\n'
    
    ### LFF_v11_0
    #general_prompt = "Solve the problem and explain your reasoning step-by-step." + extractor
    ### LFF_v11_1
    #general_prompt = "Explain your reasoning step-by-step." + extractor
    ### LFF_v11_2
    #reconsider_step = "Reconsider the conditions of question: " + question
    #reasoning_steps = [reconsider_step] + reasoning_steps
    
    ### LFF_v11_3
    #question = question + " Check that each reasoning step follows the conditions of the problem."
    
    #general_input_text = general_prompt + ' \n\n' + ' \n\n\n\n'.join(reasoning_steps) + ' \n\n\n\n'
    #general_input_text = question + ' \n\n' + ' \n\n\n\n'.join(reasoning_steps) + ' \n\n\n\n'
    
    ### LFF_v11_5
    #cut_reasoning = QAs['cut_A1']['content']   
    #cut_reasoning_steps = [l.strip() for l in cut_reasoning.split("\n") if l.strip()]
    #general_input_text = question + ' \n\n' + ' \n\n\n\n'.join(cut_reasoning_steps) + ' \n\n\n\n'
    
    ### LFF_v11_7
    general_reasoning_steps = []
    for i, step in enumerate(reasoning_steps):
        if i == 0:
            general_reasoning_steps.append("Based on problem, next step is: " + step)
        else:
            general_reasoning_steps.append(" Based on problem and/or previous reasoning, next step is: " + step)
    general_input_text = question + ' \n\n' + ' \n\n\n\n'.join(general_reasoning_steps) + ' \n\n\n\n'
    
    with torch.no_grad():
        question_input = torch.tensor([prm_tokenizer.encode(question_input_text)]).to(prm.device)
        question_logits = prm(question_input).logits[:,:,candidate_tokens]  
        general_input = torch.tensor([prm_tokenizer.encode(general_input_text)]).to(prm.device)
        general_logits = prm(general_input).logits[:,:,candidate_tokens]

        question_scores = question_logits.softmax(dim=-1)[:,:,1]
        general_scores = general_logits.softmax(dim=-1)[:,:,1]

        question_step_scores = question_scores[question_input == 23535]
        question_step_probs  = question_step_scores.tolist()
        general_step_scores = general_scores[general_input == 23535]
        general_step_probs  = general_step_scores.tolist()
    
    
    ### LFF_v11_4, LFF_v11_6
    """
    with torch.no_grad():
        question_input = torch.tensor([prm_tokenizer.encode(question_input_text)]).to(prm.device)
        question_logits = prm(question_input).logits[:,:,candidate_tokens]  

        question_scores = question_logits.softmax(dim=-1)[:,:,1]
        
        question_step_scores = question_scores[question_input == 23535]
        question_step_probs  = question_step_scores.tolist()
        
    general_step_probs = []
    for i, step in enumerate(reasoning_steps):
        ### LFF_v11_4
        #if i == 0:
        #    general_input_text = question + ' \n\n' + "Based on problem, next step is: " + step + ' \n\n\n\n'
        #else:
        #    general_input_text = question + ' \n\n' + ' \n\n\n\n'.join(reasoning_steps[:i]) + ' \n\n\n\n' + " Based on problem and previous reasoning, next step is: " + step + ' \n\n\n\n'
            
        ### LFF_v11_6
        if i == 0:
            general_input_text = question + ' \n\n' + "Based on problem, next step is: " + step + ' \n\n\n\n'
        else:
            general_input_text = question + ' \n\n' + ' \n\n\n\n'.join(reasoning_steps[:i]) + ' \n\n\n\n' + " Based on problem and/or previous reasoning, next step is: " + step + ' \n\n\n\n'
        with torch.no_grad():
            general_input = torch.tensor([prm_tokenizer.encode(general_input_text)]).to(prm.device)
            general_logits = prm(general_input).logits[:,:,candidate_tokens]
            
            general_scores = general_logits.softmax(dim=-1)[:,:,1]

            general_step_scores = general_scores[general_input == 23535]
            general_step_probs.append(general_step_scores.tolist()[-1])
    """

    QAs['step_probs1'] = question_step_probs
    QAs['step_probs2'] = general_step_probs
    
    vf1, indices1 = find_smaller_than_T(question_step_probs, threshold)
    vf2, indices2 = find_smaller_than_T(general_step_probs, threshold)
    
    QAs['vf1'] = vf1
    QAs['indices1'] = indices1
    QAs['vf2'] = vf2
    QAs['indices2'] = indices2
    
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
    set_seed(42)
    flag = 3

    threshold = 0.6  # Set the threshold for PRM

    sample_portion = 1.0  # Set the portion of the dataset (use line 104, 105 and remove line 106)
    dataset = 'GSM8K'
    model_name = "Llama-3-8B-Instruct"
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    prm_path = "UW-Madison-Lee-Lab/Llama-PRM800K"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device2 = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} and {device2}")
    
    if flag==1 or flag==3:
        ### Load the inference model
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HUGGINGFACE_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        model.eval()
        
        ### Load the prm
        candidate_tokens = [12, 10]
        prm_tokenizer = AutoTokenizer.from_pretrained(prm_path)
        prm_tokenizer.pad_token = prm_tokenizer.eos_token
        prm_tokenizer.padding_side = 'left' 
        prm_tokenizer.truncation_side = 'left'
    
        prm = AutoModelForCausalLM.from_pretrained(
            prm_path,
            torch_dtype=torch.bfloat16,
            device_map=device2
        )
        prm.eval()
    
    input_dir = "/drive2/ryusejong/LFF/dataset"
    output_dir = "/drive2/ryusejong/LFF/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path_input = f'{input_dir}/{dataset}_train_seed42_portion0.1.jsonl'
    #path_input = f'{input_dir}/{dataset}_test_seed42_portion0.1.jsonl'
    #path_input = f'{input_dir}/{dataset}_test.jsonl'
    path_output = f'{output_dir}/{dataset}_{model_name}_LFF_v11_7_train_512_cut_seed42_portion0.1.jsonl'
    #path_output = f'{output_dir}/{dataset}_{model_name}_LFF_v11_1_test_512_seed42_portion0.1.jsonl'
    #path_output = f'{output_dir}/{dataset}_{model_name}_LFF_v10_6_test_512.jsonl'

    if flag==1 or flag==3:
        data = read_data(path_input)
        sample_indices = list(np.arange(len(data)))    # use the whole dataset
             
        print(f"data size: {len(sample_indices)}, output: {path_output}, OpenAI's key: {openai.api_key}, HuggingFace's token: {HUGGINGFACE_TOKEN}")
        for i in tqdm(sample_indices): 
            messages = main(i, data[i], model, tokenizer, prm, prm_tokenizer, threshold)
            save_result(messages, path_output)
            
    if flag==2 or flag==3:
        data_est = read_data(path_output)
        length = len(data_est)
        count_1 = 0 # The accuracy of zero shot CoT prompt.
        for i in range(length):
            if data_est[i]['answer'] == data_est[i]['pred_ans']:
                count_1 += 1

        print(f"The accuracy of LFF Prompt: {count_1/length*100}.")
        path_txt = f'{output_dir}/{dataset}_{model_name}_LFF_v11_7_train_512_cut_seed42_portion0.1.txt'
        #path_txt = f'{output_dir}/{dataset}_{model_name}_LFF_v11_1_test_512_seed42_portion0.1.txt'
        #path_txt = f'{output_dir}/{dataset}_{model_name}_LFF_v10_6_test_512.txt'
        save_result_to_txt(model_name, dataset, "LFF v11", count_1/length*100, path_txt)
        
    print(f"Revision List: {len(revision_list)}, {revision_list}")
