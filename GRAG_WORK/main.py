from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from prompts.TERM_BASE_QA import extract_prompt, schema_prompt, sc_prompt, entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
# from prompts.WikiMQA import extract_prompt, schema_prompt, sc_prompt, entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
import ast
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from utils.GraphRAG import GraphRAG
# from utils.GraphRAG_v2 import GraphRAG
# from utils.KGC import KGC
# from utils.KGC_refine import KGCRefine
from utils.visualization import KgVisualize
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import argparse
import pygraphviz as pgv
from pyvis.network import Network
import random
import os
import jsonlines
from peft import PeftModel
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_question', type=str, default='/workspace/term_basev2_1203/KG_RAG_WORK/src/GraphRAG/dataset/TERM_BASE_QA/')
    parser.add_argument('--dataset_triples', type=str, default='/workspace/term_basev2_1203/KG_RAG_WORK/src/GraphRAG/dataset/all_triples/')
    parser.add_argument('--dataset_output', type=str, default='/workspace/term_basev2_1203/KG_RAG_WORK/src/GraphRAG/results/')
    # parser.add_argument('--dataset', type=str, default='TERM_BASE_QA')
    parser.add_argument('--encoder', type=str, default='/workspace/bge-large-zh-v1.5')
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--model_path', type=str, default='/workspace/Qwen2.5-7B-Instruct')
    parser.add_argument('--use_lora', type=bool, default=True)

    args = parser.parse_args()
    question_path = f"{args.dataset_question}question_0327.json"
    triples_path = f"{args.dataset_triples}combine_test_triples_0326_v2_1.json"
    output_path = f"{args.dataset_output}test_0411.json"

    print('---加载triples')
    with open(triples_path, 'r', encoding='utf-8') as f:
        triples_data = json.load(f)
    # graphRAG = GraphRAG(encoder, model,  tokenizer, question, triples_data)
    # milvus连接term_basev2_1203/KG_RAG_WORK/src/GraphRAG/dataset/all_triples/milvus.db
    connections.connect(uri="/workspace/term_basev2_1203/KG_RAG_WORK/src/GraphRAG/dataset/all_triples/milvus.db")
    col = Collection("embedding_demo", consistency_level="Strong")
    col.load()   # 必须加载到内存才能操作[4](@ref)
    refined_triplets_list = triples_data

    
    print('---加载question')
    with open(question_path) as f:
        dataset = json.load(f)
    index_size = len(dataset)
    print('index_size:',index_size)
    # print('dataset:',dataset)
    encoder = SentenceTransformer(args.encoder)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)  

    print('---数据、模型加载完毕---')
    # if args.use_lora:
    #     ft_model = PeftModel.from_pretrained(model, "dpo/results/checkpoint-984")

    # kgc = KGC(model, tokenizer, encoder)
    # kgcRefine = KGCRefine(model, tokenizer, encoder)

    def pipeline(i, refined_triplets_list, col):
        question = dataset[i]["question"]
        answer = dataset[i]["answer"]
        
        print('\n question:',question)
        # context = dataset[i]["context"]
        texts_all = []
        text2triples = {}
        # text2topic_idx = {}
        refined_res = []
        # for topic, texts in context:
        #     texts_all += texts
        #     for idx, text in enumerate(texts):
        #         text2topic_idx[text] = (topic, idx)
        
        # graphRAG = GraphRAG(encoder, model, ft_model, tokenizer, question, refined_triplets_list, triple2text)
        graphRAG = GraphRAG(encoder, model,  tokenizer, question, refined_triplets_list, col)
        # graphRAG = GraphRAG(encoder, model,  tokenizer, question, refined_triplets_list)
        # 记录开始时间
        start_time1 = time.time()
        extract_entity, topic_entity = graphRAG.TopicEntity()
        # extract_entity, topic_entity = graphRAG.dense_search()
        # 计算并输出时间差
        end_time1 = time.time()
        execution_time1 = end_time1 - start_time1
        print(f"相似实体匹配执行时间: {execution_time1:.4f} 秒")  # 保留4位小数
        # print("extract_entity: ", extract_entity)
        print("topic_entity: ",topic_entity)
        start_time2 = time.time()
        subgraph = graphRAG.SubgraphRetrieval(topic_entity)
         # 计算并输出时间差
        end_time2 = time.time()
        execution_time2 = end_time2 - start_time2
        print(f"子图查找执行时间: {execution_time2:.4f} 秒")  # 保留4位小数
        # print("subgraph: ", subgraph)
        start_time3 = time.time()
        pruned_subgraph = graphRAG.Prune(extract_entity, subgraph)
        end_time3 = time.time()
        execution_time3 = end_time3 - start_time3
        print(f"子图剪枝执行时间: {execution_time3:.4f} 秒")  # 保留4位小数
        if not pruned_subgraph:
            # simplified_q, pred = graphRAG.answering(subgraph)
            pred = graphRAG.answering(subgraph)
            # res = {"idx": i, "question": question, "simlified question": simplified_q,"answer": answer, "pred": pred, "extract entity": extract_entity, "topic entity": topic_entity, "retrieved_triples": subgraph, "supporting_facts": dataset[i]["supporting_facts"]}
            # res = {"idx": i, "question": question, "answer": answer, "pred": pred, "extract entity": extract_entity, "topic entity": topic_entity, "retrieved_triples": subgraph, "supporting_facts": dataset[i]["supporting_facts"]}
            res = {"idx": i, "question": question, "answer": answer, "pred": pred, "extract entity": extract_entity, "topic entity": topic_entity, "retrieved_triples": subgraph}
        else:    
            # simplified_q, pred = graphRAG.answering(pruned_subgraph)
            # res = {"idx": i, "question": question, "simlified question": simplified_q,"answer": answer, "pred": pred, "extract entity": extract_entity, "topic entity": topic_entity, "retrieved_triples": pruned_subgraph, "supporting_facts": dataset[i]["supporting_facts"]}
            pred = graphRAG.answering(pruned_subgraph)
            res = {"idx": i, "question": question, "answer": answer, "pred": pred, "extract entity": extract_entity, "topic entity": topic_entity, "retrieved_triples": pruned_subgraph}
            
        # except:
        #     res = {"idx": i, "question": question}

        return res 
    print('---处理数据---')
    
    if args.index != -1:
        print(f"处理第d{args.index}条t数据")
        res = pipeline(args.index)
        print(res)
    
    else:
        # 检查文件是否存在
        if os.path.exists(output_path):
            with open(output_path) as f:
                length = len(f.readlines())
        else:
            length = 0
    
        res = []

        with open(output_path, "a") as f:
            writer = jsonlines.Writer(f)

            for i, data in tqdm(enumerate(dataset[length: ]), total=len(dataset[length: ])):
                # result = pipeline(i+length,refined_triplets_list)
                result = pipeline(i+length, refined_triplets_list, col)
                print('answer:',result)
                # pipeline(i)
                res.append(result)
                writer.write(result)