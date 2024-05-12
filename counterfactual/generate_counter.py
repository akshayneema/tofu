from vllm import LLM, SamplingParams
# from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer, AutoTokenizer
from tqdm import tqdm
import torch
import re
import argparse
import pandas as pd
import uuid
from datetime import datetime
import datasets
seed=42
import transformers
from datasets import Dataset


from datasets import load_dataset
def self_vllmgen(model,df,input_field='text',output_field='text',max_tokens=512, temperature=0.7,top_p=0.9):
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature,top_p = top_p)
    # model_inputs = [tokenizer(n_shotprompt, return_tensors="pt")]
    outputs = model.generate(df[input_field].values.tolist(), sampling_params)
    answers=[output.outputs[0].text.strip() for output in outputs]
    df[output_field]=answers
    return df

def extract_responses(answer):
    prompts = []
    while True:
        pattern = f"<response>"
        start = answer.find(pattern)
        if start == -1:
            break
        end = answer.find("</response>")
        if end == -1:
            break
        prompts.append(answer[start + len(pattern):end])
        answer = answer[end + len("</task>"):]
    return prompts

def extract_counters(text):
    # This pattern matches text between <response> and </response> tags
    pattern = r"<counter>(.*?)</counter>"
    # Using re.findall to find all occurrences that match the pattern
    responses = re.findall(pattern, text, re.DOTALL)
    return responses[:4]

# model_name="/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/monday_m1"
# base_model="mistralai/Mistral-7B-v0.1"
# model_name="""/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/final_m1/merged_model"""
# model_name="""meta-llama/Meta-Llama-3-8B"""
model_name="""meta-llama/Meta-Llama-3-8B-Instruct"""

# model_name="""/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/better_m2/better-m2-filt-rand-ep5-15-April-2024-rand-dpo/merged_model"""
# model_name="""/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/better_m2/better-m2-filter-14-April-2024-ifd-dpo/merged_model"""
# model_name="""/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/better_m1/merged_modelm1"""
# model_name="""/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/final_m2/mistral-m2-filter-ep5-08-April-2024ifd-dpo/merged_model"""
# model_name="""mistralai/Mistral-7B-v0.1"""
# tokenizer_path="/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/final_m2/mistral-m2-filter-08-April-2024ifd-dpo"
# model_id="""better_m2-ep5-rand"""
# model_id="""b"""

model = LLM(model=model_name,
        tokenizer=model_name, 
        tensor_parallel_size=torch.cuda.device_count(), 
        seed=seed, 
        max_model_len=1024,
        gpu_memory_utilization=0.9, 
        dtype=torch.float16,
        
)
# pipeline = transformers.pipeline(
#     "text-generation", model=model_name, model_kwargs={"torch_dtype": torch.float16}, device_map="auto")

# print(f"############### Model/VLLM engine Loaded from {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


ncandidate=4

# output_path='./counterfactual_forget01.json'
output_path='./test.json'
split='forget01'
dataset = load_dataset("locuslab/TOFU", split)
data=dataset['train']

counterfactual_prompt="""
Given text has different facts about fictional authors. A counterfactual of a sentence is created by changing 1 fact about the author from it, keeping the rest of the sentence same . \n
Generate 4 counterfactual sentences, between <counter> tags. Text:\n
"""
def chat_template_vllm(prompt):
    prompt['chat']=f"{counterfactual_prompt} {prompt['answer']}\n\n Output:\n"
    return prompt
data=data.map(chat_template_vllm)

data=data.map(chat_template_vllm)
df=pd.DataFrame(data)
# df=df.iloc[:50]
# df.iloc[:10].to_json('./sample.json')
# print(df.head())
# for c in tqdm(range(ncandidate)):
df=self_vllmgen(model,df,input_field='chat',output_field=f'counterfactual',max_tokens=256, temperature=0.2,top_p=0.9)
df['counters']=df['counterfactual'].apply(extract_counters)

# df=self_vllmgen(model,df,input_field='chat',output_field=f'response',max_tokens=1024, temperature=0.1,top_p=0.9)
# na_rows = df_counter[df_counter.isna().any(axis=1)
# df['counterfactuals']=df['response'].apply(extract_responses)
def create_dpo_structure(row):
    entries = []
    # Assuming 'responses' is a list of four sentences
    for response in row['counters']:
        entries.append({
            'prompt': row['question'],
            'chosen': response + tokenizer.eos_token,
            'rejected': row['answer'] + tokenizer.eos_token
        })
    return entries

processed_entries = []
df.apply(lambda row: processed_entries.extend(create_dpo_structure(row)), axis=1)
dataset=Dataset.from_list(processed_entries)
dataset.to_json(f"../data/dpo-{split}.json")            

