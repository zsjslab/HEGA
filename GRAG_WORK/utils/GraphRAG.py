from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# from prompts.WikiMQA import topic_entity_prompt
from prompts.TERM_BASE_QA import topic_entity_prompt
import ast
import re
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)

class GraphRAG:
    def __init__(self, encoder, model, tokenizer, question, KG, col):
                       # encoder, model,  tokenizer, question, col
    # def __init__(self, encoder, model, ft_model, tokenizer, question, KG, triple2text):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.question = question
        # self.triple2text = triple2text
        self.col = col

        kg = nx.DiGraph()
        entities, relations = set(), set()
        for triples in KG:
            for triple in triples:
                try:
                    kg.add_edge(triple[0], triple[2], relation=triple[1])
                    entities.add(triple[0])
                    entities.add(triple[2])
                    relations.add(triple[1])
                except:
                    print(triple, '\n')
        self.graph = kg
        self.entities = entities
        self.relations = relations
        # self.ft_model = ft_model

    def generating(self, prompt, model):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        if model == self.model:
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # else:
        #     generated_ids = self.ft_model.generate(
        #         model_inputs.input_ids,
        #         max_new_tokens=512,
        #         pad_token_id=self.tokenizer.eos_token_id
        #     )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    
    def dense_search(self,query):
        limit=3
        query_dense_embedding = self.encoder.encode(str(query),normalize_embeddings=True)
        search_params = {
            "metric_type": "IP", # 内积需配合归一化使用
            "params": {
                "score_threshold": 0.7  # 过滤置信度低的结果
            }
        }
        # 执行搜索
        res = self.col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

    
    def Similarity(self, word, words):
        emb = self.encoder.encode(str(word),normalize_embeddings=True)
        similarity = {}
        for w in words: 
            e = self.encoder.encode(str(w))
            sim = cos_sim(emb, e)
            if sim > 0.7:
                similarity[w] = sim
        return similarity

    def TopicEntity(self):
        # prompt = f"""请提取出下面问题的主题实体，不要有额外的解释说明：
        # 问题: {self.question}
        # 主题实体: """
        prompt = topic_entity_prompt.prompt(self.question)
        preds = self.generating(prompt, self.model)
        try:
            preds = ast.literal_eval(preds)
        except:
            preds_ = []
            for pred in preds[1:-1].split(', '):
                preds_.append(pred[1:-1])
            preds = preds_
        tes = []
        for te in preds:
            dense_results = self.dense_search(te)
            
            # similarity = self.Similarity(te, self.entities)
            # topic_entities = sorted(similarity, key=lambda x: similarity[x], reverse=True)
            # if len(topic_entities) > 3:
            #     tes += topic_entities[:3]
            # else:
            #     tes += topic_entities
            tes += dense_results
        return preds, tes

    def SubgraphRetrieval(self, topic_entities, hop=2):
        subgraph = []
        visited = set()
        for te in topic_entities:
            centers = [(te, 0)]

            triples = []
            while centers:
                center_node, depth = centers.pop(0)
                # print("center_node: ", center_node)
                visited.add(center_node)
                if depth < hop:
                    neighbors_r = list(self.graph.neighbors(center_node))
                    neighbors_l = list(self.graph.predecessors(center_node))
                    # print('neighbors: ', neighbors)
                    for neighbor in neighbors_r:
                        if neighbor in visited:
                            continue
                        relation = self.graph.get_edge_data(center_node, neighbor)['relation']
                        if [neighbor, relation, center_node] not in triples:
                            triples.append([center_node, relation, neighbor])
                        centers.append((neighbor, depth + 1))
                    for neighbor in neighbors_l:
                        if neighbor in visited:
                            continue
                        relation = self.graph.get_edge_data(neighbor, center_node)['relation']
                        if [center_node, relation, neighbor] not in triples:
                            triples.append([neighbor, relation, center_node])
                        centers.append((neighbor, depth + 1))
            subgraph += triples
        return subgraph

    # def Prune(self, extract_entities, subgraph):
    #     pruned_subgraph = []
    #     question_no_te = self.question
    #     for te in extract_entities:
    #         question_no_te = question_no_te.replace(str(te), '')
        
    #     q_emb_0 = self.encoder.encode(question_no_te)
    #     q_emb_1 = self.encoder.encode(self.question)
    #     for triple in subgraph:
    #         tri_emb = self.encoder.encode(', '.join(triple))
    #         rel_emb = self.encoder.encode(triple[1])
    #         sim_tri = cos_sim(q_emb_1, tri_emb)
    #         sim_rel = cos_sim(q_emb_0, rel_emb)
    #         if sim_tri > 0.5 or sim_rel > 0.4:
    #             pruned_subgraph.append(triple)
    #     return pruned_subgraph
    def Prune(self, extract_entities, subgraph):
        prompt = f"""您将获得一份三元组列表，格式为[实体1, 关系, 实体2]。您的任务是从中抽取与给定问题相关的三元组。抽取的三元组应提供能够直接或间接帮助回答问题的信息。请仅输出三元组列表，无需包含任何解释或歉意。
注：必须保留‘包含’关系的三元组，避免被剪枝。
Question: {self.question}
Triples: {subgraph}
Relevant triples: """
        
#         prompt = f"""You are provided with a list of triples in the format [subject, relation, object]. Your task is to extract and return the triples that are relevant to the given question. The extracted triples should provide direct or indirect information that helps answer the question. Please just output a triple list and do not include any explanation or apologies.
# Question: {self.question}
# Triples: {subgraph}
# Relevant triples: """
        # pruned_triples = self.generating(prompt, self.ft_model)
        pruned_triples = self.generating(prompt, self.model)
        # print("pruned_triples: ", pruned_triples)
        try:
            pruned_subgraph = ast.literal_eval(pruned_triples)   
        except:
            try:
                pruned_subgraph = ast.literal_eval(pruned_triples.replace('\n', ', ')) 
            except:
                pruned_subgraph = []
                pattern = r'\[(.*?)\]' 
                output_triples = re.findall(pattern, pruned_triples)
                for text in output_triples:
                    pruned_subgraph.append(text.split(', '))

        return pruned_subgraph

    # def rag(self, triples):
    #     texts = {}
    #     for tri in triples:
    #         try:
    #             text = self.triple2text[str(tri)]
    #         except:
    #             continue
    #         if text not in texts:
    #             texts[text] = 0
    #         texts[text] += 1
    #     texts_ = sorted(texts.items(), key=lambda x: x[1], reverse=True)[:4]

    #     context = ""
    #     for i, (text, _) in enumerate(texts_):
    #         context += f"{i+1}. {text}\n"

    #     prompt = f"""Please answer the question based on the knowledge from the following texts, provide only the answer words, and do not include any explanation or apologies:
    #     Question: {self.question}
    #     Triplets: {context}
    #     Answer: """
    #     ans = self.generating(prompt, self.model)
    #     return ans

    def answering(self, pruned_subgraph):
        prompt = f"""请根据以下三元组中的知识，回答问题,不要有额外的解释或说明：
        问题: {self.question}
        三元组: {pruned_subgraph}
        答案: """
        # prompt = f"""Please answer the question based on the knowledge from the following triplets, provide only the answer, and do not include any explanation or apologies. If you cannot infer the answer to the question from the provided triples, output 'unknown'.:
        # Question: {self.question}
        # Triplets: {pruned_subgraph}
        # Answer: """

        ans = self.generating(prompt, self.model)
        # if ans == 'unknown':
        #     ans = self.rag(pruned_subgraph)
        return ans