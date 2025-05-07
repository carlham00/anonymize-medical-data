import torch
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines.token_classification import TokenClassificationPipeline

transformers.is_torch_available()

model_checkpoint = "Davlan/bert-base-multilingual-cased-ner-hrl"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# pipe = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="simple", stride=10)
# ents = pipe("Bernard works at BNP Paribas in Paris.")

pipe = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="simple", stride=10)

def anonymize(text):
    ents = pipe(text)
    split_text = list(text)
    for ent in ents:
        split_text[ent['start']] = f"[{ent['entity_group']}]"
        for i in range(ent['start'] + 1, ent['end']):
            split_text[i] = ""

    return "".join(split_text)


text = "Bernard works at BNP Paribas in Paris."
# Open the file in read mode
file = open("./transcripts/audio_file_english.txt", "r")

# Read the entire content of the file
content = file.read()
text = content
anonymized_text = anonymize(text)
print(anonymized_text)