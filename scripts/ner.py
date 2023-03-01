from transformers import pipeline

nlp = pipeline("ner", model="distilbert-base-uncased")

text = "My favorite city is New York, but I also love Paris."

entities = nlp(text)
for entity in entities:
    print(entity["word"], entity["entity"])

