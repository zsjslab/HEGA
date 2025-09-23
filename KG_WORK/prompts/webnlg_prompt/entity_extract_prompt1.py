# 1
def prompt(text):
    entity_extract_prompt = f"""Your task is to extract entities from the given text. In your answer, please strictly only include the entities and do not include any explanation or apologies.

Here are some examples:    

eg 1:
Input_chunk: Airey Neave started his career on the 30th June 1953, and ended it on the 30th March 1979.
entity_list: ['Airey_Neave', '1953-06-30', '1979-03-30']

eg 2:
Input_chunk: Egg Harbor Township is in New Jersey, in the United States and is where Atlantic City International Airport is located. This airport has a runway length of 3048.0.
entity_list: ['Egg_Harbor_Township,_New_Jersey', 'New_Jersey','Atlantic_City_International_Airport', '3048.0', 'Egg_Harbor_Township,_New_Jersey', 'United_States']

eg 3:
Input_chunk: American Abraham A. Ribicoff died in the United States and was married to Casey Ribicoff who was born in Chicago. Native Americans are one of several ethnic groups in the U.S.	
entity_list: ['Abraham_A._Ribicoff', 'Casey_Ribicoff', 'Chicago', 'United_States', 'Native_Americans_in_the_United_States', 'American']

Now, extract entities that meet the criteria from the following text. Please ensure that your answer contains only the list of entities, without any additional explanations or apologies.
Example output:['e1','e2','e3',...]

Input_chunk: {text}
entity_list:"""
    return entity_extract_prompt