import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


def retrieve_revision_advice(question, fail_memories, revision_memory, model, tokenizer, threshold=0.7):
    inputs = tokenizer(
        question,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states[-1] (type: tuple): last layer, shape [1, seq_len, dim]
    last_hidden = outputs.hidden_states[-1]  # [batch=1, seq_len, dim]
    query_vec = last_hidden[:, -1, :].squeeze(0)        # [dim]

    # compute cosine similarity between query vector and fail_memories vectors
    #query_expand = query_vec.unsqueeze(0).expand_as(fail_memories)  # [num_vector, dim]
    #cos_sims = F.cosine_similarity(fail_memories, query_expand, dim=1)  # [num_vector]
    
    # 1) dot product
    dot = fail_memories @ query_vec          # [num_vector]
    fail_mem_norm = fail_memories.norm(dim=1)     # [num_vector]
    qry_norm = query_vec.norm()       # scalar
    # 3) cosine similarity
    cos_sims = dot / (fail_mem_norm * qry_norm + 1e-8)

    max_sim, idx = torch.max(cos_sims, dim=0)
    if max_sim.item() < threshold:
        return ""
    else:
        return " Here are some advice to solve the question: " + revision_memory[idx.item()]["advice"]
    
    
# retrieve with duplication
def retrieve_error_pattern(query, fail_memories, revision_memory, retriever, threshold=0.7, num_select=1):
    
    with torch.no_grad():
        # jinaai
        #query_vec = retriever.encode(query, task="text-matching")
        # Grit
        #query_vec = retriever.encode(query, instruction=gritlm_instruction(""))
        #query_vec = retriever.encode(query, instruction=gritlm_instruction("What types of errors occur in reasoning that result in incorrect answers?"))
        # ReasonIR
        #query_vec = retriever.encode(query, instruction="")
        query_vec = retriever.encode(query, instruction="What types of errors occur in reasoning that result in incorrect answers?")
        query_vec = torch.tensor(query_vec, dtype=torch.bfloat16).to(fail_memories.device)
    
    # 1) dot product
    dot = fail_memories @ query_vec          # [num_embedding]
    fail_mem_norm = fail_memories.norm(dim=1)     # [num_embedding]
    qry_norm = query_vec.norm()       # scalar
    cos_sims = dot / (fail_mem_norm * qry_norm + 1e-8)
    
    k = min(num_select, cos_sims.size(0))
    top_sims, top_idx = torch.topk(cos_sims, k=k, largest=True)
    
    idx_list = []
    sims = []
    error_patterns = []
    for sim, idx in zip(top_sims.tolist(), top_idx.tolist()):
        if sim < threshold:
            continue
        idx_list.append(idx)
        sims.append(sim)
        error_patterns.append({
            "error_tag": revision_memory[idx]["error_tag"],
            "instruction": revision_memory[idx]["instruction"]
        })

    return idx_list, sims, error_patterns


# retrieve without duplication
def retrieve_error_pattern2(query, fail_memories, revision_memory, retriever, threshold=0.7, num_select=1):
    
    with torch.no_grad():
        # jinaai
        #query_vec = retriever.encode(query, task="text-matching")
        # Grit
        #query_vec = retriever.encode(query, instruction=gritlm_instruction("What types of errors occur in reasoning that result in incorrect answers?"))
        # ReasonIR
        query_vec = retriever.encode(query, instruction="What types of errors occur in reasoning that result in incorrect answers?")
        query_vec = torch.tensor(query_vec, dtype=torch.bfloat16).to(fail_memories.device)
    
    # 1) dot product
    dot = fail_memories @ query_vec          # [num_embedding]
    fail_mem_norm = fail_memories.norm(dim=1)     # [num_embedding]
    qry_norm = query_vec.norm()       # scalar
    cos_sims = dot / (fail_mem_norm * qry_norm + 1e-8)

    ranked_idx = torch.argsort(cos_sims, descending=True)        # [num_memory]

    idx_list = []
    sims = []
    error_patterns = []
    seen_tags = set()

    for idx in ranked_idx.tolist():
        sim_val = cos_sims[idx].item()
        if sim_val <= threshold:     
            break

        tag = revision_memory[idx]["error_tag"]
        if tag in seen_tags:      
            continue

        seen_tags.add(tag)
        idx_list.append(idx)
        sims.append(sim_val)
        error_patterns.append({
            "error_tag": tag,
            "instruction": revision_memory[idx]["instruction"]
        })

        if len(error_patterns) >= num_select:
            break

    return idx_list, sims, error_patterns
