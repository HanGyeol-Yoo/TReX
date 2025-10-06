from tokenizers import Tokenizer, pre_tokenizers, Regex
from tokenizers import normalizers, pre_tokenizers, processors, decoders
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from transformers import AutoTokenizer

def train_or_extend_tokenizer(text_files: str, vocab_size: int = 100000, setting: int = 0):
    print(f"Setting {str(setting)}!!!")

    if setting == 0:
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(
            show_progress=True, 
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            vocab_size=vocab_size,
        )

        tokenizer.normalizer = normalizers.NFC()
        
        pretokenizers = [
            pre_tokenizers.Split(
                pattern=Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                trim_offsets=False,
                use_regex=False,
            ),
        ]
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pretokenizers)

        tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
        tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)

    tokenizer.train(text_files, trainer)
    return tokenizer