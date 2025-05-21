from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("obi/deid_bert_i2b2")
model = AutoModelForTokenClassification.from_pretrained("obi/deid_bert_i2b2")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
def load_grascco_data():
    file = open("./grascco_texts/Albers.txt", "r")

    grassco_text = file.read()
    return grassco_text

def chunk_text(text, max_tokens=512, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap  # sliding window with overlap

    return chunks

example = load_grascco_data()
chunks = chunk_text(example, max_tokens=250)  # be conservative with token limit


all_results = []
for chunk in chunks:
    print(len(chunk))
    chunk_results = nlp(chunk)
    all_results.extend(chunk_results)


with open("example.ann", "w") as f:
    for i, ent in enumerate(all_results):
        print(ent)
        f.write(f"T{i+1}\t{ent['entity_group']} {ent['start']} {ent['end']}\t{ent['word']}\n")