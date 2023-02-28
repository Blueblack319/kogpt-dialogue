import json
from tqdm import tqdm
import argparse
import glob
import logging

from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)

data_src = [
    "band",
    "facebook",
    "instagram",
    "kakao1",
    "kakao2",
    "kakao3",
    "kakao4",
    "nateon",
]


def extract_from_raw(args) -> None:
    logger.info("Starting to extract data from raw!")
    for src in data_src:
        path_src = f"{args.raw_dir}/{src}/*"
        path_failed = f"{args.failed_dir}/{src}_failed.json"
        path_dst = f"{args.data_dir}/{src}.json"
        path_exceptions = f"{args.exceptions_dir}/{src}_exceptions.json"

        dataset = []
        files_failed = []
        exceptions = set()
        files = glob.glob(path_src)

        for file in tqdm(files):
            try:
                with open(file, "r", encoding="UTF-8") as f:
                    data = json.load(f)
                text = data["info"][0]["annotations"]["text"].replace("키키", "")
                dialogues = text.split("\n")
                for i, d in enumerate(dialogues):
                    dialogues[i] = d.split(":")
                    dialogues[i][0] = dialogues[i][0].strip()
                    dialogues[i][1] = dialogues[i][1].strip()
                for dialogue in dialogues:
                    speakers = set()
                    for utter in dialogue:
                        speakers.add(utter[0])
                    if len(speakers) < 3:
                        dataset.append(dialogues)
            except Exception as e:
                exceptions.add(str(e))
                logger.info(f"{file} is failed to extract...")
                files_failed.append(file)

        with open(path_dst, "w", encoding="UTF-8") as f:
            json.dump(dataset, f, ensure_ascii=False)

        with open(path_failed, "w") as f:
            json.dump(files_failed, f)

        with open(path_exceptions, "w") as f:
            json.dump(list(exceptions), f)


def load_data(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_data(tokenizer, args):
    logger.info("Starting to save the data converted into ids")
    for src in data_src:
        logging.info(f"Processing {src} data...")
        tokens_path = f"{args.data_dir}/{src}.json"
        ids_path = f"{args.data_dir}/{src}_ids.json"
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
        "--raw_dir",
        type=str,
        required=True,
        default=None,
        help="The name of the parent directory where raw files are stored.",
    )
    parser.add_argument(
        "--failed_dir",
        type=str,
        default="/home/intern/nas2/jhoon/kogpt-dialogue/failed",
        help="The name of the directory where the name of files that failed to be extracted are stored.",
    )
    parser.add_argument(
        "--exceptions_dir",
        type=str,
        default="/home/intern/nas2/jhoon/kogpt-dialogue/exceptions",
        help="The name of the directory where the exceptions are stored.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="extract",
        required=True,
        help="Extract data from raw or save ids",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__file__)

    if args.mode == "extract":
        extract_from_raw(args)
    elif args.mode == "save":
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
        save_data(tokenizer, args)
