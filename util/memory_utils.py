import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


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
