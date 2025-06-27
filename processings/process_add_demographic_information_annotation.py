import json
import os

from tqdm import tqdm


# remove first two directory
def remove_first_two_dirs(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) <= 2:
        return "/".join(parts)
    return "/".join(parts[2:])


def keep_last_dirs(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) <= 2:
        return "/".join(parts)
    return "/".join(parts[2:])


def parse_age(age):
    # Because some age are like [36Y]
    if isinstance(age, list):
        if not age:
            return None
        age = age[0]
        # if list, get the first one
    if isinstance(age, str) and age.endswith("Y"):
        try:
            return int(age[:-1])
        # only return digit part
        except ValueError:
            return None
    if isinstance(age, (int, float)):
        return age
    # return age
    return None


subdatasets = {"chexpert_full", "vindr", "isic2020", "ham10000", "pad_ufes_20", "hemorrhage", "COVID-BLUES", "cmmd"}
# dataset
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "data")
# need to change this part, only fit on my computer
metadata_lookup = {}
for dataset in subdatasets:
    entries = []
    for fname in ["annotation_train.jsonl", "annotation_test.jsonl"]:
        # Warning: Some file names for test are valid, remember to change it
        fpath = os.path.join(base_path, dataset, fname)
        # file path
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    # append each line to a big dictionary
    metadata_lookup[dataset] = entries


# add it into the dictionary of dataset
def get_demo_info(dataset, image_path):
    entries = metadata_lookup.get(dataset, [])
    target = remove_first_two_dirs(image_path)
    # remove first two
    for entry in entries:
        if "images" in entry:
            if any(target == img for img in entry["images"]):
                if dataset == "chexpert_full":
                    return {"sex": entry.get("sex"), "age": parse_age(entry.get("age"))}
                elif dataset == "vindr":
                    return {"age": parse_age(entry.get("age"))}
                elif dataset == "ham10000":
                    return {"age": parse_age(entry.get("age")), "sex": entry.get("sex")}
                elif dataset == "hemorrhage":
                    return {"age": parse_age(entry.get("age")), "gender": entry.get("gender")}
                elif dataset == "COVID-BLUES":
                    return {"age": parse_age(entry.get("age"))}
                elif dataset == "cmmd":
                    return {"age": parse_age(entry.get("age")), "race": entry.get("race")}
            elif "images" in entry and entry["images"] and os.path.basename(image_path) in entry["images"][0]:
                if dataset == "isic2020":
                    return {"age": parse_age(entry.get("age")), "sex": entry.get("sex")}
                elif dataset == "pad_ufes_20":
                    return {
                        "father": entry.get("father"),
                        "mother": entry.get("mother"),
                        "age": parse_age(entry.get("age")),
                        "gender": entry.get("gender"),
                    }

    return None


def enrich_jsonl(jsonl_filename):
    input_file = os.path.join(base_path, jsonl_filename)
    # change based on you
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
    new_data = []
    for item in tqdm(data, desc=f"Processing {jsonl_filename}"):
        # I like tqdm
        item.pop("demo", None)
        dataset = item.get("dataset")
        media = item.get("images", []) or item.get("videos", [])
        if dataset in subdatasets and media:
            demo = get_demo_info(dataset, media[0])
            if demo:
                item["demo"] = ", ".join(f"{k}: {v}" for k, v in demo.items())
        new_data.append(item)

    with open(input_file, "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")


enrich_jsonl("geom_train_upsampled.jsonl")
# enrich_jsonl("geom_valid_mini.jsonl")
