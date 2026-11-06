from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

REPO = "bakrianoo/jabarti-llm-dataset"
MAX_CHUNKS = 20
HD_FILES = {
    "phase1_train": "pretrain/phase1_train-00000-of-00001.parquet",
    "phase1_eval": "pretrain/phase1_eval-00000-of-00001.parquet",
    "phase2_train": "pretrain/phase2_train-00000-of-00001.parquet",
    "phase2_eval": "pretrain/phase2_eval-00000-of-00001.parquet",
    "ft_train": "finetune/train-00000-of-00001.parquet",
    "ft_eval": "finetune/eval-00000-of-00001.parquet",
}

TRAIN_SPLITS = {"phase1_train", "phase1_train"}
OUT_DIR = Path(__file__).parent / "output"


def download_and_filter():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in HD_FILES.items():
        local = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=filename)
        df = pd.read_parquet(local)

        if name in TRAIN_SPLITS:
            before_filter = len(df)
            df = (
                df.sort_values(["article_id", "chunk_index"])
                .groupby("article_id", sort=False)
                .head(MAX_CHUNKS)
                .reset_index(drop=True)
            )
            after_filter = len(df)

            print(f"df:{name}=>{before_filter} filteres to be {after_filter}")

        output_path = OUT_DIR / f"{name}_filtered.parquet"
        df.to_parquet(output_path, index=False)


def main():
    download_and_filter()


if __name__ == "__main__":
    main()