from sentence_transformers.util import cos_sim
import re
import ast
import random
# from prompts.WikiMQA import entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
from prompts.TERM_BASE_QA import entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
from vllm import LLM, SamplingParams
from openai import OpenAI
class KGCRefine:
    def __init__(self, model, tokenizer, encoder, use_vllm, use_api_models):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.use_vllm = use_vllm
        self.use_api_models = use_api_models
            
    def generating(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        if self.use_vllm:
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=2048)
            output = self.model.generate([text], sampling_params)
            return output[0].outputs[0].text
            
        elif self.use_api_models:
            client = OpenAI(
                # #将这里换成你在aihubmix api keys拿到的密钥
                api_key="",
                # 这里将官方的接口访问地址，替换成aihubmix的入口地址
                base_url=""
            )
            
            
            try:
                chat_completion = client.chat.completions.create(
                messages=messages,
                # model="deepseek-ai/DeepSeek-V3",
                model="DeepSeek-V3",
            )
                first_choice = chat_completion.choices[0]
                message_content = first_choice.message.content
                # print('22message_content',message_content)
                return message_content
            except Exception as e:
                print(f"API调用异常: {str(e)}")
                return ""
        else:
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


    def entityExtract(self, topic, text):
        prompt = entity_extract_prompt.prompt(topic, text)
        entities = self.generating(prompt).replace('\n',',').replace('，',',')
        if "[" not in entities:
            entities = entities.split(', ')
            return entities
        try:
            entities = ast.literal_eval(entities)
        except:
            try:
                entities = list(re.search(r'\[(.*?)\]', entities).group(1).replace("'", "").split(', '))
            except:
                
                print("error___extract entities: ", entities, "text: ", text)
        return entities

    def entityMerge(self, text, previous_entities, extracted_entities):
        prompt = entity_merge_prompt.prompt(text, previous_entities, extracted_entities)
        merged_entities = self.generating(prompt).replace('，',',').replace('\n',',')
        if "[" not in merged_entities:
            merged_entities = merged_entities.split(', ')
            return merged_entities
        try:
            merged_entities = ast.literal_eval(merged_entities)
        except:
            try:
                merged_entities = list(re.search(r'\[(.*?)\]', merged_entities).group(1).replace("'", "").split(', '))
            except:
                print("error__merged entities: ", merged_entities, "text: ", text)
        return merged_entities

    def relevantRelations(self, input_text_str, previous_relations, topk=10):
        text_embedding = self.encoder.encode(input_text_str)
        similarity_dic = {}
        for rel in previous_relations:
            rel_embedding =  self.encoder.encode(rel)
            sim = cos_sim(text_embedding, rel_embedding)
            if sim > 0.5:
                similarity_dic[rel] = sim
        if len(similarity_dic) < 10:
            relevant_relations = list(similarity_dic.keys())
        else:
            relevant_relations = [rel for rel, sim in
                                  sorted(similarity_dic.items(), key=lambda item: item[1], reverse=True)][:topk]
        return relevant_relations

    def extractTriples(self, topic, input_text, entity_hint_str, relation_hint_str):
        prompt = extract_optim_prompt.prompt(input_text, topic, entity_hint_str, relation_hint_str)
        triples = self.generating(prompt)
        try:
            triples = ast.literal_eval(triples)
            triples_ = []
            for triple in triples:
                if len(triple) == 3:
                    triples_.append(triple)
            # print('extractTriples_try:',triples_)
            # return triples_
            return self.post_process_triples(triples_)
        except:
            res = []
            triples_ = triples[1:-1].split("], ")
            for tri in triples_:
                try:
                    tri_ = ast.literal_eval(tri + ']')
                    if len(tri_) == 3: 
                        # res.append(tri_)
                        for item in tri_:
                            item = item.replace('"', '').replace(' ', '')
                            if item != "" and item != '无':
                                res.append(tri_)
                except:
                    tri = tri[1:].split(', ')
                    tri_ = []
                    for n in tri:
                        n = n.strip("'").strip('"').strip()  # 去除多余的引号和空格
                        # n = n[1:-1].replace("'", "")
                        tri_.append(n)
                    if len(tri_) == 3:
                        for item in tri_:
                            item = item.replace('"', '').replace(' ', '')
                            if item != "" and item != '无':
                                res.append(tri_)
            # print('extractTriples_except:',res)
            return self.post_process_triples(res)
            
    def post_process_triples(self, triples):
        """
        后处理三元组：
        1. 去重
        2. 删除实体名中的首尾中括号
        """
        unique_triples = []
        seen = set()  # 用于存储已见过的三元组，避免重复
    
        for triple in triples:
            # 删除实体名中的首尾中括号
            h, r, t = triple
            h = h.strip('[]')  # 去掉头实体首尾的中括号
            t = t.strip('[]')  # 去掉尾实体首尾的中括号
    
            # 构造新的三元组
            new_triple = [h, r, t]
    
            # 检查是否已经存在
            triple_tuple = tuple(new_triple)  # 转换为元组以便存入集合
            if triple_tuple not in seen:
                seen.add(triple_tuple)
                unique_triples.append(new_triple)
    
        return unique_triples
                

    def construct_refinement_hint(self,text2topic_idx, input_text_list, extracted_triplets_list, relations, include_relation_example="self",
                                  relation_top_k=10):

        entity_hint_list = []
        relation_hint_list = []

        relation_example_dict = {}
        if include_relation_example == "self":
            # Include an example of where this relation can be extracted
            for idx in range(len(input_text_list)):
                input_text_str = input_text_list[idx]
                extracted_triplets = extracted_triplets_list[idx]
                for triplet in extracted_triplets:
                    relation = triplet[1]
                    if relation not in relation_example_dict:
                        relation_example_dict[relation] = [{"text": input_text_str, "triplet": triplet}]
                    else:
                        relation_example_dict[relation].append({"text": input_text_str, "triplet": triplet})
        else:
            # Todo: allow to pass gold examples of relations
            pass

        # print("________start constructing hint lists______")
        for idx in range(len(input_text_list)):
            input_text_str = input_text_list[idx]
            extracted_triplets = extracted_triplets_list[idx]

            previous_relations = set()
            previous_entities = set()

            for triplet in extracted_triplets:
                if len(triplet) == 3:
                    previous_entities.add(triplet[0])
                    previous_entities.add(triplet[2])
                    previous_relations.add(triplet[1])

            previous_entities = list(previous_entities)
            previous_relations = list(previous_relations)

            # Obtain candidate entities
            topic = text2topic_idx[input_text_str][0]
            extracted_entities = self.entityExtract(topic, input_text_str)
            # print("(1) extrated entities: ", extracted_entities)

            merged_entities = self.entityMerge(
                input_text_str, previous_entities, extracted_entities
            )
            # print("(2) merged entities: ", merged_entities)

            entity_hint_list.append(str(merged_entities))

            # Obtain candidate relations
            hint_relations = previous_relations

            retrieved_relations = self.relevantRelations(input_text_str, previous_relations)

            counter = 0

            for relation in retrieved_relations:
                if counter >= relation_top_k:
                    break
                else:
                    if relation not in hint_relations:
                        hint_relations.append(relation)

            candidate_relation_str = ""
            for relation_idx, relation in enumerate(hint_relations):
                if relation not in relations.keys():
                    continue

                relation_definition = relations[relation]

                candidate_relation_str += f"{relation_idx + 1}. {relation}: {relation_definition}\n"

                if include_relation_example == "self":
                    if relation not in relation_example_dict:
                        # candidate_relation_str += "Example: None.\n"
                        pass
                    else:
                        selected_example = None
                        if len(relation_example_dict[relation]) != 0:
                            selected_example = random.choice(relation_example_dict[relation])
                        # for example in relation_example_dict[relation]:
                        #     if example["text"] != input_text_str:
                        #         selected_example = example
                        #         break
                        if selected_example is not None:
                            candidate_relation_str += f"Example: '{selected_example['triplet']}' can be extracted from '{selected_example['text']}'\n"
                            # candidate_relation_str += f"""例如,{selected_example['triplet']}可以从"{selected_example['text']}中提取"\n"""
                        else:
                            # candidate_relation_str += "Example: None.\n"
                            pass
            relation_hint_list.append(candidate_relation_str)
            # print("(3) relation hint list: ", relation_hint_list)
        return entity_hint_list, relation_hint_list