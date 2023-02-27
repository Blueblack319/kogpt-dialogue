import json
import glob
import argparse
from tqdm import tqdm


def extract_from_raw(args) -> None:
    dataset = []
    files_failed = []
    files = glob.glob(args.src_path)

    for file in tqdm(files):
        try:
            with open(file, "r", encoding="UTF-8") as f:
                data = json.load(f)
                if data["info"][0]["annotations"]["speaker_type"] == "1:1":
                    dials = data["info"][0]["annotations"]["text"].replace("키키", "")
                    data_splitted = dials.split("\n")
                    for i, d in enumerate(data_splitted):
                        data_splitted[i] = d.split(" : ")
                        data_splitted[i][1] = data_splitted[i][1].strip()
                    dataset.append(data_splitted)
        except:
            files_failed.append(file)

    with open(args.dst_path, "w") as f:
        json.dump(dataset, f)

    with open(args.files_failed, "w") as f:
        json.dump(files_failed, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        default=None,
        required=True,
        help="The source path for pre-processing data",
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        default=None,
        required=True,
        help="The destination path to save preprocessed data",
    )
    parser.add_argument(
        "--files_failed",
        type=str,
        default=None,
        required=True,
        help="The path for saving failed files",
    )
    args = parser.parse_args()
    extract_from_raw(args)
