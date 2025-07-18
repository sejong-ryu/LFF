import openai
import json 
import time 
import re
import random
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" 
from tqdm.auto import tqdm
from util.utils import set_seed, read_data, save_result, get_answer_from_text, chat_huggingface, save_result_to_txt, save_result_to_txt2
from util.memory_utils import retrieve_revision_advice, retrieve_error_pattern, retrieve_error_pattern2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


openai.api_key = "xxx"
HUGGINGFACE_TOKEN = "xxx"

revision_list = []

"""_summary_
    - Question + response의 vector cos similarity를 기준으로 가장 유사한 ERROR memory를 찾아서, 해당 ERROR memory의 error_tag과 instruction을 반환.
    - Instruction는 \"extract_failure_latent\" 파일의 \"New Error Extract Prompt\"로 GPT-o3에서 추출 (모든 failure case을 대상으로 GPT-o3를 통해 대표되는 error_tag와 instruction을 추출).
    - Prompt 주는 방식: CoT prompt + " You often make the " + error_tag + " when reasoning. To avoid this error:" + instruction + "Review your previous answer and re-solve the problem."
"""

def main(i, data, model, tokenizer=None, retriever=None, fail_memories=None, revision_memories=None):
    QAs = dict()
    QAs['index'] = int(i)

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
    
    ### Extract embedding from question + zero-shot CoT prompt + extractor
    # jinaai
    #query = data["question"] + "\n" + response_1 + "\n" + "What types of errors occur in reasoning that result in incorrect answers?"
    # Grit
    query = data["question"] + "\n" + response_1
    idx_list, sims, error_patterns = retrieve_error_pattern2(query, fail_memories, revision_memories, retriever, threshold=0.6, num_select=1)
    
    if len(error_patterns) != 0:
        
        QAs['idx_list'] = idx_list
        QAs['sims'] = sims
        QAs['error_pattenrs'] = error_patterns
        
        question = f"You often make the "
        avoid_prompt = "To avoid this error:\n"
        for i, error_pattern in enumerate(error_patterns):
            question += f"{i+1}.\"{error_pattern['error_tag']}\" "
            avoid_prompt += f"\t{i+1}. {error_pattern['instruction']}\n"
        question = question + "when reasnoning.\n" + avoid_prompt + "Review your previous answer and re-solve the problem."
        #print("-"*50)
        #print(question)
        
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

    sample_portion = 1.0  # Set the portion of the dataset (use line 104, 105 and remove line 106)
    #fail_memory_path = "memory/FAILUREs_reasonir.pt"
    fail_memory_path = "memory/FAILURE_ERRORs_reasonir.pt"
    revision_memory_path = "memory/ERRORs.jsonl"
    dataset = 'GSM8K'
    model_name = "Llama-3-8B-Instruct"
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    retriever_path = "reasonir/ReasonIR-8B"
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
        )
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        model.eval()
        model = model.to(device)
        
        ### Load the retriever 
        retriever = AutoModel.from_pretrained(
            retriever_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device2,
        )
        retriever.eval()
    
    input_dir = "dataset/"
    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #path_input = f'{input_dir}/{dataset}_train.jsonl'
    path_input = f'{input_dir}/{dataset}_test_seed42_portion0.1.jsonl'
    path_output = f'{output_dir}/{dataset}_{model_name}_LFF_v8_9_test_512_seed42_portion0.1.jsonl'

    if flag==1 or flag==3:
        fail_memories = torch.load(fail_memory_path, map_location=device).to(torch.bfloat16).to(device)
        print(f"fail memories: {fail_memories.shape}")
        revision_memories = read_data(revision_memory_path)
        print(f"finished load memoreis")
        data = read_data(path_input)
        
        #sample_k = int(np.floor(sample_portion * len(data)))
        #sample_indices = np.random.choice(np.arange(len(data)), size=sample_k, replace=False)
        sample_indices = list(np.arange(len(data)))    # use the whole dataset
             
        print(f"data size: {len(sample_indices)}, output: {path_output}, OpenAI's key: {openai.api_key}, HuggingFace's token: {HUGGINGFACE_TOKEN}")
        for i in tqdm(sample_indices): 
            messages = main(i, data[i], model, tokenizer, retriever, fail_memories, revision_memories)
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
        #path_txt = f'{output_dir}/{dataset}_{model_name}_LFF_train.txt'
        path_txt = f'{output_dir}/{dataset}_{model_name}_LFF_v8_9_test_512_seed42_portion0.1.txt'
        save_result_to_txt(model_name, dataset, "LFF v8", count_1/length*100, path_txt)
        
    print(f"Revision List: {len(revision_list)}, {revision_list}")
