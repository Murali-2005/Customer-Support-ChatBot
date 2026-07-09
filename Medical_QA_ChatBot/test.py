from medical_langchain_helper import (
    get_medical_response,
    clear_history
)

while True:

    question = input("Ask: ")

    if question.lower() == "exit":
        break

    if question.lower() == "clear":
        clear_history()
        print("History Cleared\n")
        continue

    answer = get_medical_response(question)

    print("\nAnswer:\n")
    print(answer)
    print("-" * 80)