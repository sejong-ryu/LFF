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
