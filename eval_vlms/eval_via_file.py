import ujson
from evaluation import compute_metrics_by_data_source


def eval_via_file(file_path):
    """
    Evaluate a VLM using a file-based approach.
    This function reads the input data from a file, processes it, and evaluates the VLM.
    """
    with open(file_path, "r") as f:
        data = ujson.load(f)["results"]

    predictions, ground_truths, data_sources, datasets = [], [], [], []

    for item in data:
        predictions.append(item["model_response"]["content"])
        ground_truths.append(item["ground_truth"])
        data_sources.append(item["data_source"])
        datasets.append(item["dataset"])

    # Compute metrics
    metrics = compute_metrics_by_data_source(predictions, ground_truths, data_sources, datasets)


if __name__ == "__main__":
    eval_via_file("results/o4-mini_results.json")
