import openai
import json 
import time 
import re
import random
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5" 
from tqdm.auto import tqdm
from util.utils import set_seed, read_data, save_result, get_answer_from_text, chat_huggingface, save_result_to_txt, save_result_to_txt2
from util.memory_utils import retrieve_revision_advice, retrieve_error_pattern, retrieve_error_pattern2
from util.prm_utils import find_smaller_than_T
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


openai.api_key = "xxx"
HUGGINGFACE_TOKEN = "xxx"

revision_list = []

def main(i, data, model, tokenizer=None, prm=None, threshold=0.6):
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
    
    QAs['A1'] = {'role': 'assistant', 'content':response_1}
    messages.append({'role': 'assistant', 'content':response_1})
    
    QAs['answer'] = answer
    QAs['pred_ans1'] = get_answer_from_text(response_1)
    
    ### PRM
    question = QAs['Q1']['content'].strip()
    reasoning = QAs['A1']['content']   
    
    reasoning_steps = [l.strip() for l in reasoning.split("\n") if l.strip()]
    #prm_input_text = question + ' \n\n' + ' \n\n\n\n'.join(reasoning_steps) + ' \n\n\n\n'
    
    ### LFF_v11_1
    general_prompt = "Explain your reasoning step-by-step." + extractor
    prm_input_text = general_prompt + ' \n\n' + ' \n\n\n\n'.join(reasoning_steps) + ' \n\n\n\n'
    
    with torch.no_grad():
        prm_input = torch.tensor([tokenizer.encode(prm_input_text)]).to(prm.device)
        prm_logits = prm(prm_input).logits[:,:,candidate_tokens]
        #print(logits.shape)
        prm_scores = prm_logits.softmax(dim=-1)[:,:,1]
        #print(scores.shape)
        step_scores = prm_scores[prm_input == 23535]
        step_probs  = step_scores.tolist()
        
    QAs['step_probs'] = step_probs
    #vf, indices = find_smaller_than_T1(step_probs, threshold)
    vf, indices = find_smaller_than_T(step_probs, threshold)
    
    ### Second inference depending on PRM
    if vf:
        #mistake_steps = reasoning_steps[indices[0]]
        mistake_steps = [reasoning_steps[index] for index in indices]
        
        #question = f"Review your previous answer, especially \"{mistake_steps}\". Re-solve the problem."
        
        # LFF_v9_4
        """
        question = f"Review your previous answer, especially \"{mistake_steps}\". Solve the problem again in a totally different way. Ensure that each step directly uses only the numbers and conditions from the current problem, matching them one-to-one. Count the conditions and numbers in the problem, and count them again in your reasoning to ensure nothing is missing or added. Always track what each number represents, avoid double-counting or unnecessary calculations, and confirm that your final answer directly answers what the problem asks." + extractor
        """
        
        # LFF_v9_6
        """
        question = f"Review your previous answer. Solve the problem again in a totally different way. Ensure that each step directly uses only the numbers and conditions from the current problem, matching them one-to-one. Count the conditions and numbers in the problem, and count them again in your reasoning to ensure nothing is missing or added. Always track what each number represents, avoid double-counting or unnecessary calculations, and confirm that your final answer directly answers what the problem asks." + extractor
        """
        # LFF_v9_7
        """
        question = f"Review your previous answer. Solve the problem again in a totally different way." + extractor
        """
        
        # LFF_v9_8
        """
        question = f"Review your previous answer, especially \"{mistake_steps}\". Solve the problem again in a totally different way." + extractor
        """
        
        # LFF_v9_5
        """
        question = f"Review and analyze your previous answer deeply, especially \"{mistake_steps}\". Maintaining the previous reasoning steps, update your answer based on your review and analysis and re-solve the problem." + extractor
        """
        
        # LFF_v9_9
        
        step_examples = ", ".join([f"\"{step}\"" for step in mistake_steps])
        question = f"Review your previous answer, especially " + step_examples + " Solve the problem again in a totally different way. Ensure that each step directly uses only the numbers and conditions from the current problem, matching them one-to-one. Count the conditions and numbers in the problem, and count them again in your reasoning to ensure nothing is missing or added. Always track what each number represents, avoid double-counting or unnecessary calculations, and confirm that your final answer directly answers what the problem asks." + extractor
        
        
        # LFF_v9_10
        """
        step_examples = ", ".join([f"\"{step}\"" for step in mistake_steps])
        question = "Review your previous answer, especially " + step_examples + " Solve the problem again in a totally different way." + extractor
        """
        
        # LFF_v9_11
        """
        step_examples = ", ".join([f"\"{step}\"" for step in mistake_steps])
        question = "Review your previous answer, especially " + step_examples + " Solve the problem again in a different way." + extractor
        """
        
        QAs['Q2'] = {'role': 'user', 'content': question}
        messages.append({'role': 'user', 'content': question})
        
        response_2 = chat_huggingface(messages, model, tokenizer, max_new_tokens=512)
        
        QAs['A2'] = {'role': 'assistant', 'content':response_2}
        messages.append({'role': 'assistant', 'content':response_2})
        
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
        prm_tokenizer.pad_token = tokenizer.eos_token
        prm_tokenizer.padding_side = 'left' 
        prm_tokenizer.truncation_side = 'left'
    
        prm = AutoModelForCausalLM.from_pretrained(
            prm_path,
            torch_dtype=torch.bfloat16,
            device_map=device2
        )
        prm.eval()
    
    input_dir = "/drive2/ryusejong/LFF/dataset"
    output_dir = '/drive2/ryusejong/LFF/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path_input = f'{input_dir}/{dataset}_train_seed42_portion0.1.jsonl'
    #path_input = f'{input_dir}/{dataset}_test_seed42_portion0.1.jsonl'
    #path_input = f'{input_dir}/{dataset}_test.jsonl'
    path_output = f'{output_dir}/{dataset}_{model_name}_LFF_v12_1_9_train_512_seed42_portion0.1.jsonl'
    #path_output = f'{output_dir}/{dataset}_{model_name}_LFF_v12_1_9_test_512_seed42_portion0.1.jsonl'
    #path_output = f'{output_dir}/{dataset}_{model_name}_LFF_v9_6_test_512.jsonl'

    if flag==1 or flag==3:
        data = read_data(path_input)
        sample_indices = list(np.arange(len(data)))    # use the whole dataset
             
        print(f"data size: {len(sample_indices)}, output: {path_output}, OpenAI's key: {openai.api_key}, HuggingFace's token: {HUGGINGFACE_TOKEN}")
        for i in tqdm(sample_indices): 
            messages = main(i, data[i], model, tokenizer, prm, threshold)
            save_result(messages, path_output)
            
    if flag==2 or flag==3:
        data_est = read_data(path_output)
        length = len(data_est)
        count_1 = 0 # The accuracy of zero shot CoT prompt.
        for i in range(length):
            if 'pred_ans2' in data_est[i]:
                pred = data_est[i]['pred_ans2']
            else:
                pred = data_est[i]['pred_ans1']
            if data_est[i]['answer'] == pred:
                count_1 += 1

        print(f"The accuracy of LFF Prompt: {count_1/length*100}.")
        path_txt = f'{output_dir}/{dataset}_{model_name}_LFF_v12_1_9_train_512_seed42_portion0.1.txt'
        #path_txt = f'{output_dir}/{dataset}_{model_name}_LFF_v9_11_test_512_seed42_portion0.1.txt'
        #path_txt = f'{output_dir}/{dataset}_{model_name}_LFF_v9_6_test_512.txt'
        save_result_to_txt(model_name, dataset, "LFF v12", count_1/length*100, path_txt)
        
    print(f"Revision List: {len(revision_list)}, {revision_list}")
