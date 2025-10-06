from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing, Sequence
from tokenizers import AddedToken
import click
import json
import os

@click.command()
@click.option(
    "--process_tgt",
    type=str,
)
@click.option(
    "--output_dir",
    type=str,
)
def main(
    process_tgt: str,
    output_dir: str,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    new_tok = PreTrainedTokenizerFast.from_pretrained(process_tgt)

    new_tok.add_tokens(
        AddedToken(
            content="<|BOS|>",
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=True, 
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|EOS|>",
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=True, 
            special=True
        )
    )
    
    new_tok.bos_token = "<|BOS|>"
    new_tok.eos_token = "<|EOS|>"

    new_tok._tokenizer.post_processor = Sequence(
        [
            new_tok._tokenizer.post_processor, 
            TemplateProcessing(
                single=f"{new_tok.bos_token} $A",
                pair=f"{new_tok.bos_token} $A {new_tok.bos_token} $B",
                special_tokens=[
                    (new_tok.bos_token, new_tok.bos_token_id),
                ]
            )
        ]
    )
    
    new_tok.save_pretrained(output_dir)

    new_tok.model_input_names = ['input_ids', 'attention_mask']
    
    with open(os.path.join(output_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    update_data = {
        "model_input_names": [
            "input_ids",
            "attention_mask"
        ],
    }

    config.update(update_data)

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()