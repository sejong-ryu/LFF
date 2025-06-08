# LFF

## Latent Failure Feedback (LFF) Experiment

### Process

1. **Initial Inference**  
   Performed inference on the GSM8K *train* dataset using the `run_GSM8K_zeroshot_CoT.py` script and saved the outputs.

2. **Latent Extraction from Incorrect Samples**  
   Extracted the final hidden (latent) vector of each incorrectly answered question using the `extract_failure_latent.ipynb` notebook.  
   Generated revised prompts for those failure cases using GPT-o3.

3. **Latent-Guided Prompt Injection**  
   In the `run_GSM8K_LFF.py` script, each test question's latent vector was compared with the failure latent vectors using cosine similarity (threshold = 0.5).  
   - If a similar failure case was found, its corresponding revised prompt was injected before inference.  
   - If no match was found, the original question was used as-is.

### Result

- **Accuracy** **73.24% → 72.93%**

### Notes

- Indices of test samples modified with revised prompts are recorded in `test_revision_list.txt`.
