from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
# from prompts.TERM_BASE_QA import extract_prompt, schema_prompt, sc_prompt, entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
import ast
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# from utils.GraphRAG import GraphRAG
# from utils.KGC import KGC
# from utils.KGC_refine import KGCRefine
# from utils.webnlg_kg_zyq import webnlg_kg_zyq
from utils.domain_kg_zyq import domain_kg_zyq
# from utils.visualization import KgVisualize
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib
import argparse
# import pygraphviz as pgv
# from pyvis.network import Network
import random
import jsonlines
from vllm import LLM, SamplingParams
import pandas as pd
import time
import transformers
import torch
# 记录开始时间
start_time = time.time()

parser = argparse.ArgumentParser()
# term_basev2_1203/KG_WORK/src/GraphRAG/dataset/rebel/rebel_1000.csv
parser.add_argument('--input_path', type=str, default='/workspace/term_basev2_1203/KG_WORK/src/GraphRAG/dataset/domain_data/domain_data_0620.csv')
# 实体抽取：en_entity.json
parser.add_argument('--llm_model', type=str, default='chatgpt-4o-latest')
parser.add_argument('--llm_model_jc', type=str, default='gpt')
parser.add_argument('--encoder', type=str, default='/workspace/bge-large-zh-v1.5')
parser.add_argument('--model_path', type=str, default='/workspace/Qwen2.5-7B-Instruct')
parser.add_argument('--index', type=int, default=-1)
parser.add_argument('--use_vllm', type=bool, default=False)   
parser.add_argument('--use_api_models', type=bool, default=True) 
parser.add_argument('--output_version', type=str, default='0623') 
parser.add_argument('--label', type=str, default='last') #'no_disambiguation','no_en','no_re', 'no_en_re', 'no_en_re_pr'

args = parser.parse_args()
# output_path = "/workspace/term_basev2_1203/output/test0319.json"
# 输入和输出文件路径
input_path = args.input_path
# term_basev2_1203/KG_WORK/src/GraphRAG/dataset/rebel/result/test
output_path_pre = '/workspace/term_basev2_1203/KG_WORK/src/GraphRAG/dataset/domain_data/result/'
entity_output_path = f"{output_path_pre}entity/entity_{args.llm_model_jc}_first_{args.output_version}.json"
# entity_output_path = f"{output_path_pre}entity/entity_{args.llm_model_jc}_first_0620.json"
replace_entity_output_path = f"{output_path_pre}entity/entity_{args.llm_model_jc}_disambiguation_{args.output_version}.json"
triple_last_output_path = f"{output_path_pre}triple_0620/domain_triple_{args.llm_model_jc}_{args.label}_{args.output_version}.json"
dataset = pd.read_csv(input_path)
# dataset = dataset['text']
dataset.reset_index(drop=True, inplace=True)
# print('dataset:',dataset)
encoder = SentenceTransformer(args.encoder)
# model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto')

# ------------------------模型选择---------------------------------------------
if args.use_api_models:
    print('使用API')
    # 如果使用API模型，则不需要初始化本地model和tokenizer
    model = None
    tokenizer = None
    pipeline = None
elif args.use_vllm:
    # 否则，正常初始化
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)    
    model = LLM(model=args.model_path)
    pipeline = None
else:
    if args.llm_model_jc == 'llama':
        pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda:1",
        )
        model = None
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)  
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cuda:1')
        pipeline = None
# ---------------------------------------------------------------------
kg_zyq = domain_kg_zyq(model, tokenizer,pipeline, encoder, args.use_vllm,args.use_api_models, args.llm_model)

# ------------------------实体抽取---------------------------------------------
entity_dict = {}
if os.path.exists(entity_output_path):
    with open(entity_output_path, 'r', encoding='utf-8') as f:
        entity_dict = json.load(f)
else:
    for i, data in tqdm(enumerate(dataset['text']), total=len(dataset)):
        entity_lists = kg_zyq.entity_Extract(data)
        entity_dict[data] = entity_lists
        
    # if args.llm_model == 'chatgpt-4o-latest':
    #     # 保存为 JSON 文件
    with open(entity_output_path, 'w', encoding='utf-8') as f:
        json.dump(entity_dict, f, ensure_ascii=False, indent=4)
# # ---------------------------------------------------------------------
end_time1 = time.time()
print(f"代码运行时间1: {end_time1 - start_time:.6f} 秒")
# ------------------------聚类---------------------------------------------
clustered_entities = kg_zyq.entity_clustering(entity_dict)

# with open(f"{output_path_pre}entity/entity_{args.llm_model_jc}_cluster_{args.output_version}.json", 'w', encoding='utf-8') as f:
#     json.dump(clustered_entities, f, ensure_ascii=False, indent=4)
print('clustered_entities:',clustered_entities)
# ---------------------------------------------------------------------

# ------------------------大模型消岐---------------------------------------------
disambiguation_results = {}
for key, entity_list in tqdm(clustered_entities.items()):
    if key == -1:
        for value in tqdm(entity_list):
            disambiguation_result = kg_zyq.entity_disambiguation(value)
            print('-1_disambiguation_result:',disambiguation_result)
            # 假设 disambiguation_result 是形如 {'key': [value]} 的字典
            for canonical_name, entities in disambiguation_result.items():
                # 将结果添加到最终结果字典中
                disambiguation_results[canonical_name] = entities
        # continue  # 跳过 key 为 -1 的项
    else:
    # 执行实体消岐
        disambiguation_result = kg_zyq.entity_disambiguation(entity_list)
        print('disambiguation_result:',disambiguation_result)
        # print(entity_list,disambiguation_result)
        # 将结果追加到最终结果字典中
        for canonical_name, entity_list in disambiguation_result.items():
            disambiguation_results[canonical_name] = entity_list
    

# 执行替换
final_entity_result = kg_zyq.replace_entities(entity_dict, disambiguation_results)
# 输出结果或保存到文件
# print(json.dumps(final_result, indent=2, ensure_ascii=False))

# 可选：将结果写入新文件
with open(replace_entity_output_path, "w", encoding="utf-8") as f:
    json.dump(final_entity_result, f, indent=2, ensure_ascii=False)
end_time2 = time.time()
# print(f"代码运行时间1: {end_time1 - start_time:.6f} 秒")
print(f"代码运行时间2: {end_time2 - start_time:.6f} 秒")

# ---------------------------------------------------------------------
 
#------------------断点续传--------------------------
with open(replace_entity_output_path, "r", encoding="utf-8") as f:
    final_entity_result = json.load(f)
triple_dict = {}
relation = {}
output_path = triple_last_output_path
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        existing_data = list(jsonlines.Reader(f))
    start_keys = {item['key'] for item in existing_data}  # 已处理过的 key
else:
    existing_data = []
    start_keys = set()
# 对新数据进行处理
with jsonlines.open(output_path, mode='a') as writer:
    for i, (key, value) in tqdm(enumerate(final_entity_result.items())):
        if key in start_keys:
            continue  # 跳过已处理的 key
        triple_lists, relation = kg_zyq.triple_Extract(key, value, relation)
        # first_triple_no_en_re_pr_dict[key] = first_triple_no_en_re_pr_lists
        # 写入当前处理结果到文件
        writer.write({
            'key': key,
            'triples': triple_lists
        })
# --------------------------------------------------------------------- 

end_time3 = time.time()
# print(f"代码运行时间1: {end_time1 - start_time:.6f} 秒")
print(f"代码运行时间3: {end_time3 - start_time:.6f} 秒")










