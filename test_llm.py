from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import cassis


# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "LSX-UniWue/LLaMmlein_1B",
    device_map="auto",
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained("LSX-UniWue/LLaMmlein_1B")

# Prepare prompt and messages
# prompt = "Bitte den folgenden Text annotieren: Sehr geehrte Frau Kollegin Weigel, wir berichten über oben genannte Patientin, die sich zuletzt am 13.12.2029 in unserer Tumorsprechstunde vorstellte."


with open("/home/brendan/Downloads/Clausthal/TypeSystem.xml", "rb") as f:
    typesystem = cassis.load_typesystem(f)


with open("/home/brendan/Downloads/Clausthal/Clausthal.xmi", "rb") as f:
    cas = cassis.load_cas_from_xmi(f, typesystem)

sofa = cas.sofa_string
entity_strings = []

for phi in cas.select("custom.PHI"):
    # print(phi.kind)
    entity_strings.append(f'{{"entity": "{sofa[phi.begin:phi.end]}", "label": "{phi.kind}"}}')

entities_json = "[" + ", ".join(entity_strings) + "]"

# print(entities_json)

prompt = (
    "Bitte das folgenden Text-To-Annotate Beispiel annotieren. Ich gebe zuerst ein Beispiel Text mit den entsprechenden Annotationen dazu.\n
    "Nimm bitte das Text-To-Annotate Beispiel und gib mir bitte die Annotationen nur dafür zurück, nicht die aus dem Antwort Beispiel."
    f"Text Beispiel: '{sofa}'\n"
    f"Antwort Beispiel: '{entities_json}'\n"
    "Text-To-Annotate: 'Sehr geehrte Frau Kollegin Weigel, wir berichten über oben genannte Patientin, die sich zuletzt am 13.12.2029 in unserer Tumorsprechstunde vorstellte.'\n"
    "Antwort:"
)


# Tokenize input and move it to the GPU
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=5000,
    num_return_sequences=1,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# # # Show memory summary
# # # torch.cuda.empty_cache()
# # # print(torch.cuda.memory_summary())


