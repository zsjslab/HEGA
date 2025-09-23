from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
# from prompts.WikiMQA import extract_prompt, schema_prompt, sc_prompt, entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
from prompts.TERM_BASE_QA import extract_prompt, schema_prompt, sc_prompt, entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
import ast
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# from utils.GraphRAG import GraphRAG
from utils.KGC import KGC
from utils.KGC_refine import KGCRefine
from utils.visualization import KgVisualize
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import argparse
import pygraphviz as pgv
from pyvis.network import Network
import random
import jsonlines
from vllm import LLM, SamplingParams
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/workspace/term_basev2_1203/KG_RAG_WORK/data/chunk_data/')
    parser.add_argument('--output_path', type=str, default='/workspace/term_basev2_1203/KG_RAG_WORK/output/triples_data/')
    parser.add_argument('--encoder', type=str, default='/workspace/bge-large-zh-v1.5')
    parser.add_argument('--model_path', type=str, default='/workspace/Qwen2.5-7B-Instruct')
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--use_vllm', type=bool, default=False)   
    parser.add_argument('--use_api_models', type=bool, default=True) 
    args = parser.parse_args()
    # output_path = "/workspace/term_basev2_1203/output/test0319.json"
    # 输入和输出文件路径
    input_path = f"{args.input_path}0508_text_205.csv"
    output_path = f"{args.output_path}0508_triples_205/0509_triples_205_6_v3.json"
    
    
    dataset = pd.read_csv(input_path)
    dataset = dataset[dataset['type'] != 'figure']
    dataset.reset_index(drop=True, inplace=True)
    # print('dataset:',dataset)
    encoder = SentenceTransformer(args.encoder)
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto')
    if args.use_api_models:
        print('使用API')
        # 如果使用API模型，则不需要初始化本地model和tokenizer
        model = None
        tokenizer = None
    else:
        # 否则，正常初始化
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)    
        model = LLM(model=args.model_path)
    kgc = KGC(model, tokenizer, encoder, args.use_vllm,args.use_api_models)
    kgcRefine = KGCRefine(model, tokenizer, encoder, args.use_vllm,args.use_api_models)

    relations = {}

    def pipeline(i, relations):
        # question = dataset[i]["question"]
        # answer = dataset[i]["answer"]
        if dataset["type"][i] == 'table':
            context = dataset["tabel_figure_content"][i]
        else:
            context = dataset["content"][i]
        topic = dataset["headings"][i]
        print('i==',i)
        # print('dataset[content][i]::',dataset["content"][i])
        texts_all = [context]
        text2topic = {}
        text2topic[context] = [topic]
        # try:
        # print('texts_all:',texts_all)
        triples_lists, relations = kgc.extract(text2topic, texts_all, relations)
        entity_hint_list, relation_hint_list = kgcRefine.construct_refinement_hint(text2topic, texts_all, triples_lists,
                                                                                    relations)
        refined_triplets_list = []

        for idx, input_text in enumerate(texts_all):
            # input_text = texts_all
            print('input_text:',input_text)
            entity_hint_str = entity_hint_list[idx]
            relation_hint_str = relation_hint_list[idx]
            topic = text2topic[input_text][0]
            refined_triplets = kgcRefine.extractTriples(topic, input_text, entity_hint_str, relation_hint_str)
            if not refined_triplets:
                refined_triplets = triples_lists[idx]
            # print('refined_triplets:',refined_triplets)
            refined_triplets_list.append(refined_triplets)

        add_chunk_list = refined_triplets_list[0].copy()
        for idx, triple_set in enumerate(add_chunk_list):
            # print(idx,data)
            # 再次确保 triple_set 是一个列表并且不为空
            if isinstance(triple_set, list) and len(triple_set) == 3:
                h, r, t = triple_set  # 解包三元组
                # print('h:',h)
                triple = [context, '包含', h]
                triple_topic = [topic, '包含', h]
                if triple not in refined_triplets_list[0]:
                    # print('chunk:',triple)
                    refined_triplets_list[0].append(triple) 
                    
                if triple_topic not in refined_triplets_list[0]:
                    # print('chunk:',triple)
                    refined_triplets_list[0].append(triple_topic)
            else:
                print(f"Unexpected format for triple: {triple_set}")

        
            # print('relations:',relations)
            # print("triples:",refined_triplets_list)
            # print('\n')
        # print("once_triples",triples_lists)
        print("triples",refined_triplets_list)
        # return relations, {"once_triples": triples_lists, "last_triples": refined_triplets_list}
        return relations, {"last_triples": refined_triplets_list}

   
    # if os.path.exists(output_path):
    #     with open(output_path) as f:
    #         lens = len(f.readlines())
    #         if lens>= 1:
    #             start = lens-1
    #             print('len(f.readlines()):',lens)
    #         else:
    #             start = 0
    # else:
    #     start = 0
    # print('start:',start)
    # with open(output_path,'a') as f:
    #     writer = jsonlines.Writer(f)
        
    #     for i, data in tqdm(enumerate(dataset['content'][start:]), total=len(dataset['content'][start:])):
    #         print('total,i:',len(dataset['content'][start:]),i)
    #         relations, new = pipeline(i+start, relations)
    #         writer.write(new)
    #         print('\n')
    print('output_path:',output_path)
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_data = list(jsonlines.Reader(f))
        start = len(existing_data)-1
        print('start1111:', start)
    else:
        start = 0
        existing_data = []
        
    print('start:', start)
    # 定义一个函数来处理空的 last_triples
    def process_empty_triples(i, relations):
        while True:
            relations, new = pipeline(i, relations)
            if new.get('last_triples') != [[]]:  # 确保 last_triples 不为空
                return relations, new
    # 对新数据进行处理
    with jsonlines.open(output_path, mode='a') as writer:
        for i, data in tqdm(enumerate(dataset['content'][start:]), total=len(dataset['content'][start:])):
            current_index = i + start
            if not existing_data or (existing_data and existing_data[-1].get('last_triples') != [[]]):
                relations, new = pipeline(current_index, relations)  # 调整索引以匹配dataset中的位置
            else:
                # 如果最后一条记录是 {"last_triples": [[]]}，则重新抽取
                relations, new = process_empty_triples(current_index, relations)
            writer.write(new)

# 注意：确保pipeline函数以及dataset变量在调用process_dataset之前已经被正确初始化。
            

