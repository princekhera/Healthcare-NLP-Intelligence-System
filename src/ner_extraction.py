import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    
    entities = {
        "diseases": [],
        "drugs": [],
        "orgs": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["orgs"].append(ent.text)
        elif ent.label_ == "GPE":
            continue
        else:
            entities["diseases"].append(ent.text)
    
    return entities