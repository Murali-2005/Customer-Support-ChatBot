import os
import xml.etree.ElementTree as ET
import pandas as pd

# Path to MedQuAD dataset
DATASET_PATH = "../MedQuAD-master"

rows = []

for root_dir, _, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.endswith(".xml"):
            continue

        file_path = os.path.join(root_dir, file)

        try:

            tree = ET.parse(file_path)
            root = tree.getroot()

            for qa in root.findall("./QAPairs/QAPair"):

                question = qa.find("Question")
                answer = qa.find("Answer")

                if question is None or answer is None:
                    continue

                q = "".join(question.itertext()).strip()
                a = "".join(answer.itertext()).strip()

                if q and a:

                    rows.append({
                        "prompt": q,
                        "response": a,
                        "source": os.path.basename(root_dir)
                    })

        except Exception as e:

            print(f"Skipped {file}: {e}")

df = pd.DataFrame(rows)

output_path = "medical_dataset.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)

print("=" * 50)
print(f"Total Questions : {len(df)}")
print(f"Saved File      : {output_path}")
print("=" * 50)