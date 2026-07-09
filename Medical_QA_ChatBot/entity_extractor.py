import spacy

nlp = spacy.load("en_core_web_sm")

# Basic medical vocabulary
DISEASES = {
    "diabetes", "asthma", "cancer", "covid", "covid-19",
    "hypertension", "leukemia", "arthritis", "pneumonia",
    "stroke", "migraine"
}

SYMPTOMS = {
    "fever", "headache", "cough", "fatigue", "pain",
    "vomiting", "nausea", "dizziness", "chest pain",
    "shortness of breath", "blurred vision", "thirst",
    "weight loss"
}

TREATMENTS = {
    "chemotherapy", "radiation", "surgery",
    "insulin", "therapy", "transplant"
}

MEDICINES = {
    "ibuprofen", "paracetamol", "acetaminophen",
    "metformin", "aspirin", "amoxicillin"
}


def extract_entities(text):

    doc = nlp(text.lower())

    entities = {
        "Diseases": [],
        "Symptoms": [],
        "Treatments": [],
        "Medicines": []
    }

    # Check noun chunks (captures phrases like "chest pain")
    for chunk in doc.noun_chunks:

        value = chunk.text.strip()

        if value in DISEASES:
            entities["Diseases"].append(value)

        if value in SYMPTOMS:
            entities["Symptoms"].append(value)

        if value in TREATMENTS:
            entities["Treatments"].append(value)

        if value in MEDICINES:
            entities["Medicines"].append(value)

    # Check individual tokens
    for token in doc:

        word = token.text

        if word in DISEASES and word not in entities["Diseases"]:
            entities["Diseases"].append(word)

        if word in SYMPTOMS and word not in entities["Symptoms"]:
            entities["Symptoms"].append(word)

        if word in TREATMENTS and word not in entities["Treatments"]:
            entities["Treatments"].append(word)

        if word in MEDICINES and word not in entities["Medicines"]:
            entities["Medicines"].append(word)

    return entities