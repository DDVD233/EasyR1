import json
from collections import defaultdict


def format_reasoning():
    """
    Format the reasoning data for annotation
    """
    with open("val_generations_0.json", "r") as f:
        data = json.load(f)

    formatted_data = []
    dataset_count = defaultdict(int)
    index = 0

    important_ids = [123, 114, 94, 84, 83, 82, 81]
    modalities = []
    fill_index = 0

    # First, fill these ids with as diverse sample as possible
    for item in data:
        dataset = item["input"].split("Dataset: ")[1].split(" \n")[0]
        dataset_count[dataset] += 1

        question = item["input"].split("user\n")[1].split("You FIRST think")[0]
        modality = question.split("Above is a ")[1].split("of a patient")[0]
        if modality in modalities:
            continue
        modalities.append(modality)
        label = item["label"]
        reasoning_trace = item["output"].replace("\n", "").split("<think>")[1].split("</think>")[0]
        data_path = item["input"].split("Datapath: ")[1].split(" \n")[0]
        prediction = item["output"].split("\\boxed{")[1].split("}")[0]
        formatted_data.append(
            (
                f"ID-{important_ids[fill_index]:03d}",
                data_path,
                prediction,
                label,
                f'"Question: {question} Ground Truth Label: {label} \nReasoning to annotate: {reasoning_trace}"',
            )
        )
        fill_index += 1
        if fill_index >= len(important_ids):
            break

    for item in reversed(data):
        dataset = item["input"].split("Dataset: ")[1].split(" \n")[0]
        dataset_count[dataset] += 1

        if dataset_count[dataset] > 5:
            continue
        while index in important_ids:
            index += 1

        question = item["input"].split("user\n")[1].split("You FIRST think")[0]
        label = item["label"]
        reasoning_trace = item["output"].replace("\n", "").split("<think>")[1].split("</think>")[0]
        data_path = item["input"].split("Datapath: ")[1].split(" \n")[0]
        try:
            prediction = item["output"].split("\\boxed{")[1].split("}")[0]
        except:
            prediction = ""
        formatted_data.append(
            (
                f"ID-{index:03d}",
                data_path,
                prediction,
                label,
                f'"Question: {question} Ground Truth Label: {label} \nReasoning to annotate: {reasoning_trace}"',
            )
        )
        index += 1

    # save as annotation.csv with two columns: origin and content
    with open("annotation.csv", "w") as f:
        f.write("origin,content\n")
        for item in formatted_data:
            f.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]}\n")

    with open("annotation.json", "w") as f:
        json.dump(formatted_data, f, indent=4)


if __name__ == "__main__":
    format_reasoning()
