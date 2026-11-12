from pathlib import Path
import pandas as pd

from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
    normalizers,
)
from cleaning import build_normalizer 

#configuration

CURRENT_WD=Path(__file__).parent
CLEAN_PATHS=[
    CURRENT_WD/"output"/"phase1_train_clean.parquet",
    CURRENT_WD/"output"/"phase2_train_clean.parquet",
]
OUTPUT_PATH= CURRENT_WD/"tokenizer.json"
VOCAB_SIZE=32_000
MIN_FREQUENCY=2
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]",
                  "[AR]", "[EN]",
                  "[SYS]", "[USER]", "[ASST]"]

# Tokenizer pipeline

def build_tokenizer():
     tokenizer=Tokenizer(models.BPE(unk_token="[]"))
     tokenizer.normalizer=build_normalizer()
     tokenizer.pre_tokenizer= pre_tokenizers.Metaspace(prepend_scheme="always")
     tokenizer.decoder=decoders.Metaspace(prepend_scheme="always")
     return tokenizer
#batch by batchtokenizerMetaspace
def corpus_iterator(df, batch_size=1000):
    texts=df["text"].tolist()
    for i in range(0,len(texts),batch_size):
      yield texts[i: i+batch_size]


def train(tokenizer, df):
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        corpus_iterator(df),
        trainer=trainer,
        length=len(df),
    )

    return tokenizer




def main():
    print("Load Cleaned Corpus")
    frames = []

    for clean_path in CLEAN_PATHS:
        frame = pd.read_parquet(clean_path)
        print(f"{clean_path.name}: {len(frame)}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    print("Build Pipeline + Train BPE")
    tokenizer = build_tokenizer()
    tokenizer = train(tokenizer=tokenizer, df=df)

    print("Sanity Check: encode/decode bilingual examples")

    examples = [
        "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة",
        "The United States",
        "أحمد إبراهيم آدم علىّ",
    ]

    for text in examples:
        encoded_text = tokenizer.encode(
            text,
            add_special_tokens=True
        )

        decoded_text = tokenizer.decode(encoded_text.ids)

        print(f"tokens: {encoded_text.tokens}")
        print(f"ids: {encoded_text.ids}")
        print(f"decoded: {decoded_text}")

    tokenizer.save(str(OUTPUT_PATH))
    print(f"Tokenizer saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

