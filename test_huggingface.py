import torch
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers.pipelines.token_classification import TokenClassificationPipeline


transformers.is_torch_available()

model_checkpoint = "Babelscape/wikineural-multilingual-ner"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

pipe = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="simple", stride=10)
# ents = pipe("Bernard works at BNP Paribas in Paris.")

# pipe = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

def anonymize(text):
    ents = pipe(text)
    print(ents)
    split_text = list(text)
    for ent in ents:
        split_text[ent['start']] = f"[{ent['entity_group']}]"
        for i in range(ent['start'] + 1, ent['end']):
            split_text[i] = ""

    return "".join(split_text)

    #     nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    # split_text = list(text)
    # print(nlp)
    # # for ent in nlp:
    # #     split_text[ent['start']] = f"[{ent['entity_group']}]"
    # #     for i in range(ent['start'] + 1, ent['end']):
    # #         split_text[i] = ""

    # return "".join(split_text)


def load_grascco_data():
    file = open("./grascco_texts/Albers.txt", "r")

    grassco_text = file.read()
    return grassco_text


# text = "Bernard works at BNP Paribas in Paris."
# Open the file in read mode
# file = open("./transcripts/audio_file_english.txt", "r")

# # Read the entire content of the file
# content = file.read()
# text = content

text = load_grascco_data()
anonymized_text = anonymize(text)
print(anonymized_text)