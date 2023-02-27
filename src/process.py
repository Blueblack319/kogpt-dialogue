import json
from tqdm import tqdm
import argparse

from transformers import PreTrainedTokenizerFast


def load_data(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_data(tokenizer, args):
    tokens_path = f"{args.data_dir}/{args.tokens_file}.json"
    ids_path = f"{args.data_dir}/{args.tokens_file}_ids.json"
    dialogues = load_data(tokens_path)
    ids = []
    for dialogue in tqdm(dialogues):
        dialogue_ids = []
        for utter in dialogue:
            utter_ids = []
            tokens = tokenizer.tokenize(utter[1])
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            utter_ids.append(utter[0])
            utter_ids.append(token_ids)
            dialogue_ids.append(utter_ids)
        ids.append(dialogue_ids)

    msg = "Lengths must be equal"
    assert len(ids) == len(dialogues), msg

    with open(ids_path, "w") as f:
        json.dump(ids, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        default=None,
        help="The name of the parent directory where data files are stored.",
    )
    parser.add_argument(
        "--tokens_file",
        type=str,
        default=None,
        required=True,
        help="The name of the tokens file without extension",
    )

    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")

    save_data(tokenizer, args)
