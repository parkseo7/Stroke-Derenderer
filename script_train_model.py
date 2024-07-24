from npi_class.train import ModelTrainer
from npi_class.model.dataset import TextBWDataset, get_filepaths

from pathlib import Path
import yaml

def load_yaml(filepath):
    """Loads a .yaml file.
    """
    
    with open(filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    
    return data_loaded


def main(args):
    """Main training function.
    """
    
    args_dirs = args["directories"]
    args_model = args["model"]
    model_name = args_model["name"]
    args_train = args["train"]

    # Create model path:
    model_path = Path(args_dirs["model"]) / model_name
    model_path.mkdir(parents=True, exist_ok=True)

    # Get dataset:
    base_path = args_dirs["data"]
    args_data = args["data"]
    split = args_train["split"]
    hw_folders = args_data["handwritten"]
    pr_folders = args_data["printed"]

    data = get_filepaths(base_path, hw_folders, pr_folders, split=split)
    (train_paths, train_labels), (val_paths, val_labels) = data
    train_dataset = TextBWDataset(train_paths, train_labels)
    val_dataset = TextBWDataset(val_paths, val_labels)

    # Just use standard transform:
    print("Computing mean, standard deviation of training set...")
    mean, std = train_dataset.get_stats()
    val_dataset.update_transform(mean, std)

    print(f"Mean, std of transform is {mean}, {std}")
    mt = ModelTrainer(model_path)
    metrics, model = mt.train_model(args, train_dataset, val_dataset)

    return metrics, model


if __name__ == "__main__":
    p = "/home/daniel/npi-class/configs/config.yaml"
    args = load_yaml(p)
    metrics, model = main(args)