# 2
def prompt(text, extracted_triples, extracted_relations):
    relation_define_prompt = f"""You will be given a piece of text and a list of relational triples in the format of [Subject, Relation, Object] extracted from the text. For each relation present in the triples, your task is to write a description to express the meaning of the relation.

Here are some examples:    

eg 1:
chunk: Emily Warren, born in New York on June 23, 1890, graduated from MIT in 1914 with a B.S. in engineering. She died in Boston on January 22, 1977.
Triples: [["Emily_Warren", "birthPlace", "New_York"],["Emily_Warren", "birthDate", "June_23,_1890"],["Emily_Warren", "deathPlace", "Boston"],["Emily_Warren", "deathDate", "January_22,_1977"],["Emily_Warren", "almaMater", "MIT, B.S. 1914"]]
relations: ["birthPlace", "birthDate", "deathPlace", "deathDate", "almaMater"]
relation_define:
birthPlace: The subject entity was born in the location specified by the object entity.
birthDate: The subject entity was born on the date specified by the object entity.
deathPlace: The subject entity died in the location specified by the object entity.
deathDate: The subject entity died on the date specified by the object entity.
almaMater: The subject entity received an academic degree or education from the institution specified by the object entity, often including the degree and year.

eg 2:
chunk: Adam Koc is a national of Poland where the language used is Polish. He was involved in battles of the Polish-Soviet war during which Joseph Stalin was a commander.	
Triples: [['Adam_Koc', 'battle', 'Polish–Soviet_War'], ['Poland', 'language', 'Polish_language'], ['Polish–Soviet_War', 'commander', 'Joseph_Stalin'], ['Adam_Koc', 'nationality', 'Poland']]
relations:['battle', 'language', 'commander', 'nationality']
relation_define:
battle: The subject entity participated in the military conflict or war specified by the object entity.
language: The language specified by the object entity is spoken in the country or region represented by the subject entity.
commander: The person specified by the object entity commanded or led the subject entity during a military operation or war.
nationality: The subject entity holds the nationality of the country specified by the object entity.

eg 3:
chunk: Bacon Explosion is a dish from the Unit States, where Joe Biden is leader and the capital is Washington DC. English is the language spoken in the US and Asian Americans are an ethnic group there.
Triples: [['Bacon_Explosion', 'country', 'United_States'], ['United_States', 'leader', 'Joe_Biden'], ['United_States', 'ethnicGroup', 'Asian_Americans'], ['United_States', 'capital', 'Washington,_D.C.'], ['United_States', 'language', 'English_language']]
relations:['country', 'leader', 'ethnicGroup', 'capital', 'language']
relation_define:
country: The subject entity originates from or is associated with the country specified by the object entity.
leader: The person specified by the object entity is the head of state or government of the country represented by the subject entity.
ethnicGroup: The group specified by the object entity is an ethnic population residing within the country represented by the subject entity.
capital: The city specified by the object entity is the capital of the country represented by the subject entity.
language: The language specified by the object entity is spoken in the country represented by the subject entity.

Now please extract relation descriptions given the following text and triples. Note that the description needs to be general and can be used to describe relations between other entities as well. Pay attention to the order of subject and object entities.
Please describe according to the example and do not include any explanation or apologies.

chunk:{text}
Triples:{extracted_triples}
relations:{extracted_relations}
relation_define:"""
    return relation_define_prompt