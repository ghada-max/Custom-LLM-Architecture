
""" loads the bilingual tokenizer already trained """

from pathlib import Path
from tokenizers import Tokenizer as BackingTokenizer

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "assets" / "tokenizer.json"


class Tokenizer:
    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "[SEP]": 4,
        "[AR]": 5,
        "[EN]": 6,
        "[SYS]": 7,
        "[USER]": 8,
        "[ASST]": 9,
    }

    def __init__(self, backing_tokenizer: BackingTokenizer):
        self._tokenizer = backing_tokenizer

    @classmethod
    def from_file(cls, path=DEFAULT_TOKENIZER_PATH):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"No tokenizer at {path}."
            )

        return cls(
            BackingTokenizer.from_file(str(path))
        )

    def normalize(self, text: str) -> str:
        return self._tokenizer.normalizer.normalize_str(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(
            ids, skip_special_tokens=skip_special_tokens
        )

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
