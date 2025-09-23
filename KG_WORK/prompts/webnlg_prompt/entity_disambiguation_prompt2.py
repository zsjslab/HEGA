# 2
def prompt(entity_list):
    entity_disambiguation_prompt = f"""You are an entity disambiguation assistant. I will provide you with a list of entity names. Please determine which entities refer to the same real-world entity. without any additional explanations or apologies!
    
Here are some examples:    

eg 1:
Input list:['San_Francisco,_California', 'Auburn,_California', 'Auburn,_Alabama', 'San_Fran,_California']
output_dict:{{"San_Francisco,_California": ["San_Francisco,_California", 'San_Fran,_California'], 'Auburn,_California':['Auburn,_California'], 'Auburn,_Alabama':['Auburn,_Alabama']}}

eg 2:
Input list:['Albuquerque', 'Albuquerque,_New_Mexico']
output_dict:{{"Albuquerque,_New_Mexico": ['Albuquerque', 'Albuquerque,_New_Mexico']}}

eg 3:
Input list:['Big_Hero_6', 'Big_Hero_6_(film)']
output_dict:{{"Big_Hero_6_(film)": ['Big_Hero_6', 'Big_Hero_6_(film)']}}

Notes:
- The canonical name should be the most standard or clear version of the entity.
- If an entity is unrelated to others, it should appear as its own key-value pair.
- Do not omit any input entities.
- Only output the output_dict, do not output any other explanations or analysis!
Example output:{{"canonical_entity_name1": ["original_entity1", "original_entity2", ...],"canonical_entity_name2": ["original_entity3", "original_entity4", ...],...}}

Input list:{entity_list}
output_dict:"""
    return entity_disambiguation_prompt