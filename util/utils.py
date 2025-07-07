import random
import numpy as np
import torch
import json
import re
import openai


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        
def save_result_to_txt(model, dataset, method, accuracy, filename="result.txt"):
    content = f"Model: {model}\nDataset: {dataset}\nMethod: {method}\nAccuracy: {accuracy:.10f}"
    with open(filename, "w") as f:
        f.write(content)
        

def save_result_to_txt2(model, dataset, method, accuracy1, accuracy2, filename="result.txt"):
    content = f"Model: {model}\nDataset: {dataset}\nMethod: {method}\nStardard Accuracy: {accuracy1:.10f}\n{method} Accuracy: {accuracy2:.10f}\n"
    with open(filename, "w") as f:
        f.write(content)
        
        
def save_result_to_txt3(model, dataset, method, accuracy1, accuracy2, accuracy3, filename="result.txt"):
    content = f"Model: {model}\nDataset: {dataset}\nMethod: {method}\nStardard Accuracy: {accuracy1:.10f}\nIoE Accuracy: {accuracy2:.10f}\nFinal Accuracy: {accuracy3:.10f}\n"
    with open(filename, "w") as f:
        f.write(content)
        
        
def save_result_to_txt3(model, dataset, method, accuracy1, accuracy2, accuracy3, filename="result.txt"):
    content = f"Model: {model}\nDataset: {dataset}\nMethod: {method}\nStardard Accuracy: {accuracy1:.10f}\nIoE Accuracy: {accuracy2:.10f}\nFinal Accuracy: {accuracy3:.10f}\n"
    with open(filename, "w") as f:
        f.write(content)
        

def save_sampled_indices_to_txt(seed, length_sample_indices, sample_indices, filename="sampled_indices.txt"):
    indices_str = np.array2string(sample_indices, separator=', ', threshold=np.inf)
    content = f"Seed: {seed}\nLength: {length_sample_indices}\nIndices: {indices_str}"
    with open(filename, "w") as f:
        f.write(content)
        
        
def read_data(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data


def save_result(messages, path):
    f = open(path, 'a+')
    json.dump(messages, f)
    f.write('\n')
    f.close()
    
    
def normalize_answer(ans):
    # ans = ans.lower()
    ans = ans.replace(',', '')
    #ans = ans.replace('.', '')
    ans = ans.replace('?', '')
    ans = ans.replace('!', '')
    ans = ans.replace('\'', '')
    ans = ans.replace('\"', '')
    ans = ans.replace(';', '')
    ans = ans.replace(':', '')
    ans = ans.replace('-', '')
    ans = ans.replace('_', '')
    ans = ans.replace('(', '')
    ans = ans.replace(')', '')
    ans = ans.replace('[', '')
    ans = ans.replace(']', '')
    ans = ans.replace('{', '')
    ans = ans.replace('}', '')
    ans = ans.replace('/', '')
    ans = ans.replace('\\', '')
    ans = ans.replace('|', '')
    ans = ans.replace('<', '')
    ans = ans.replace('>', '')
    ans = ans.replace('=', '')
    ans = ans.replace('+', '')
    ans = ans.replace('*', '')
    ans = ans.replace('&', '')
    ans = ans.replace('^', '')
    ans = ans.replace('%', '')
    ans = ans.replace('$', '')
    # ans = ans.replace('#', '')
    ans = ans.replace('@', '')
    ans = ans.replace('~', '')
    ans = ans.replace('`', '')
    ans = ans.replace(' ', '')
    return ans


def get_answer_from_text(sentence):
    sentence = sentence.replace(',', '')     # To remove the punctuation in number, e.g., $2,000
    pattern = re.compile(r'##(.*?)##')
    #pattern = re.compile(r'#{2,}\s*(.*?)\s*#{2,}', re.DOTALL)

    ans = re.findall(pattern, sentence)
    if len(ans):
        ans = ans[-1]
        ans = normalize_answer(ans)
        try:
            ans = float(ans)
        except:
            ans = float(10086100100)
    else:
        ans = float(10086100100)
    return ans


def contruct_conversation(messages):
    conversation = ""
    for m in messages:
        role = m.get("role", "").lower()
        content = m.get("content", "").strip()

        if role == "system":
            conversation += content + "\n\n"
        elif role == "user":
            conversation += f"User: {content}\n\n"
        elif role == "assistant":
            conversation += f"Assistant: {content}\n\n"
        else:
            conversation += content + "\n\n"
    conversation += "Assistant:"
    return conversation
    
    
def chat_huggingface(messages, model, tokenizer, max_new_tokens=256):
    conversation = contruct_conversation(messages)
    #print("-"*50)
    #print(conversation)

    inputs = tokenizer(
        conversation,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0]
    input_len = inputs.input_ids.shape[-1]
    generated_text = tokenizer.decode(gen_ids[input_len:], skip_special_tokens=True).strip()
    return generated_text
