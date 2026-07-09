from entity_extractor import extract_entities

while True:

    text = input("Ask: ")

    print()

    print(extract_entities(text))

    print()