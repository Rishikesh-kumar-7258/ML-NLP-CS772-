# utils/data_loader.py
import os, random, subprocess
from datasets import Dataset, DatasetDict, load_dataset

def download_and_extract(data_dir="./hin"):
    if not os.path.exists(data_dir):
        print("📥 Downloading Aksharantar Hindi dataset...")
        subprocess.run(["wget", "-O", "hin.zip", "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/hin.zip"], check=True)
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(["unzip", "hin.zip", "-d", data_dir], check=True)
        os.remove("hin.zip")

def load_aksharantar_hindi(max_train=100_000, seed=42):
    download_and_extract()
    
    test = load_dataset("json", data_files="hin/hin_test.json")["train"]
    val = load_dataset("json", data_files="hin/hin_valid.json")["train"]
    train_stream = load_dataset("json", data_files="hin/hin_train.json", streaming=True)["train"]

    train_list = []
    for item in train_stream:
        src = item["english word"].strip()
        tgt = item["native word"].strip()
        if src and tgt:
            train_list.append({"source": src, "target": tgt})
        if len(train_list) >= max_train * 2:
            break

    random.seed(seed)
    random.shuffle(train_list)
    train_list = train_list[:max_train]

    def clean(ds):
        cols = [c for c in ds.column_names if c not in ["english word", "native word"]]
        if cols: ds = ds.remove_columns(cols)
        return (
            ds
            .rename_column("english word", "source")
            .rename_column("native word", "target")
            .filter(lambda x: x["source"] and x["target"])
        )

    return DatasetDict({
        "train": Dataset.from_list(train_list),
        "validation": clean(val),
        "test": clean(test)
    })