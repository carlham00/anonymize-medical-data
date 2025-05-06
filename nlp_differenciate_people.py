import spacy
from spacy.matcher import Matcher

nlp = spacy.load('de_core_news_sm')

# text = (
#     "Manchmal trifft sich Oskar mit Gunther in Sonnenbuehl. Oskar ist schon 9 Jahre alt. Gunther dagegen ist erst 7 Jahre."
# )

# doc = nlp(text)

### REDACT OR REPLACE MENU
global known_names
known_names = []
def redact_entity(entity_type, token):
    if entity_type == "NAME":
        # see if hash of name is already known to us
        # if its known just find the number of first occurence
        # if its new we add to the counter of known people
        check = hash(str(token))
        if check not in known_names:
            known_names.append(check)
            redacted_text = "[REDACTED_" + entity_type + "0" + str(len(known_names)) + "] "
        else:
            known_number = known_names.index(check)
            redacted_text = "[REDACTED_" + entity_type + "0" + str(known_number) + "] "
    else:
        # all other entities are redacted the same way and we dont differentiate
        redacted_text = "[REDACTED_" + entity_type + "] "
    return redacted_text

def replace_entity(entity_type):
    if entity_type == "NAME":
        replaced_text = "Max Mustermann"
    elif entity_type == "AGE":
        replaced_text = "ein paar Jahre alt"
    elif entity_type == "LOCATION":
        replaced_text = "Sonnenbuehl"
    return replaced_text

### FIND ELEMENTS THAT ARE SENSITIVE
### 1. NAMES
# https://realpython.com/natural-language-processing-spacy-python/
def redact_names_in_doc(nlp_doc):
    with nlp_doc.retokenize() as retokenizer:
        for ent in nlp_doc.ents:
            retokenizer.merge(ent)
    tokens = []
    for token in nlp_doc:
        if token.ent_iob != 0 and token.ent_type_ == "PER":
            tokens.append(redact_entity("NAME", token))
        else:
            tokens.append(token.text_with_ws)
    return nlp(" ".join(tokens))

### 2. AGE
# https://spacy.io/usage/rule-based-matching
# https://stackoverflow.com/questions/57395165/extracting-a-persons-age-from-unstructured-text-in-python
def redact_age_in_doc(nlp_doc):
    matcher = Matcher(nlp.vocab)

    #pattern = [{"LIKE_NUM": True},{"IS_PUNCT": True,"OP":"*"},{"LEMMA": "Jahre"}, {"IS_PUNCT": True,"OP":"*"}, {"LEMMA": "alt"},{"IS_ALPHA": True, "OP":"*"},{'POS':'PROPN',"OP":"*"},{'POS':'PROPN',"OP":"*"}]

    age_pattern = [{"IS_DIGIT": True, "OP": "+"}, 
                   {"IS_SPACE": True, "OP": "*"}, 
                   {"TEXT": "Jahre"},
                   {"IS_SPACE": True, "OP": "*"}, 
                   {"TEXT": "alt", "OP": "*"}]

    matcher.add("AGE", [age_pattern])
    matches = matcher(nlp_doc)

    schemes = []
    for i in range(0, len(matches)):
        start, end = matches[i][1], matches[i][2]
        span = nlp_doc[start:end]
        schemes.append(str(span[0]))

    with nlp_doc.retokenize() as retokenizer:
        for ent in nlp_doc.ents:
            retokenizer.merge(ent)
    tokens = []
    for token in nlp_doc:
        if str(token) in schemes:
            tokens.append(redact_entity("AGE", token))
        else:
            tokens.append(token.text_with_ws)
    return nlp("".join(tokens))

# names = redact_names_in_doc(doc)
# ages = redact_age_in_doc(names)

# print(ages)