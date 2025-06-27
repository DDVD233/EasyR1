import ujson


def visualize_multiview(annotations_path: str, output_path: str):
    data = []
    with open(annotations_path, "r") as f:
        for line in f:
            data.append(ujson.loads(line.strip()))

    print(f"Loaded {len(data)} annotations from {annotations_path}")

    for sample in data:
        if "lateral" not in sample["images"][0]:
            continue

        if "Lung Lesion" not in sample["conversations"][1]["value"]:
            continue

        print(f"Processing sample: {sample['images'][0]}")


if __name__ == "__main__":
    visualize_multiview(
        annotations_path="/scratch/high_modality/chest_xray/chexpert_full/annotation_train.jsonl",
        output_path="path/to/output",
    )
