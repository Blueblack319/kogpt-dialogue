import argparse
import logging
import os
from pprint import pformat
from itertools import chain

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_polynomial_decay_schedule_with_warmup,
)
import torchmetrics

from custom_dataset import CustomDataset, PadCollate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


class KoGPTDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(KoGPTDataModule, self).__init__()
        ### TODO: 조금 더 이쁘게 할 수 있을듯, DataModule에만 필요한 args를 따로 뽑아서 넣어주기?
        self.args = args

    def setup(self, stage=None):
        # Load train & valid dataset
        logger.info("Loading train & valid data...")
        self.train_set = CustomDataset(self.args.train_prefix, self.args)
        self.valid_set = CustomDataset(self.args.valid_prefix, self.args)
        self.ppd = PadCollate(eos_id=self.args.eos_id)
        self.num_batches = len(self.train_dataloader())

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            collate_fn=self.ppd.pad_collate,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            collate_fn=self.ppd.pad_collate,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,
        )


class KoGPTTask(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPTTask, self).__init__()
        self.save_hyperparameters(hparams)
        pl.seed_everything(self.hparams.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.hparams.model_type)
        self.model.resize_token_embeddings(self.hparams.vocab_size)
        self.hparams.max_len = min(self.hparams.max_len, self.model.config.n_ctx)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--max_len",
            type=int,
            default=1024,
            help="The maximum length of input sequence.",
        )
        parser.add_argument(
            "--warmup_ratio",
            type=float,
            default=0.1,
            help="The ratio of warmup steps to the total training steps.",
        )
        parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size.")
        return parser

    def training_step(self, batch, batch_ids):
        input_ids, token_type_ids, labels = batch
        outputs = self.model(
            input_ids=input_ids, token_type_ids=token_type_ids, labels=labels
        )
        loss = outputs[0]
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, labels = batch
        outputs = self.model(
            input_ids=input_ids, token_type_ids=token_type_ids, labels=labels
        )
        loss = outputs[0]
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        # warm up lr
        total_train_steps = self.hparams.num_batches * self.hparams.max_epochs
        warmup_steps = int(self.hparams.warmup_ratio * total_train_steps)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
            power=2,
        )
        return [optimizer], [scheduler]

    def infer(self):
        if self.hparams.gpu:
            self.hparams.device = torch.device("cuda")
        else:
            self.hparams.device = torch.device("cpu")
        self.model.to(self.hparams.device)
        with torch.no_grad():
            input_hists = []

            while True:
                utter = input("You: ")
                if utter == self.hparams.end_command:
                    print("Bot: Good bye.")
                    break
                input_ids = [self.hparams.sp1_id] + self.hparams.tokenizer.encode(utter)
                input_hists.append(input_ids)

                if len(input_hists) >= self.hparams.max_turns:
                    num_exceeded = len(input_hists) - self.hparams.max_turns + 1
                    input_hists = input_hists[num_exceeded:]
                input_ids = (
                    [self.hparams.bos_id]
                    + list(chain.from_iterable(input_hists))
                    + [self.hparams.sp2_id]
                )
                start_sp_id = input_hists[0][0]
                next_sp_id = (
                    self.hparams.sp2_id
                    if start_sp_id != self.hparams.sp2_id
                    else self.hparams.sp1_id
                )
                assert start_sp_id != next_sp_id
                token_type_ids = [
                    [start_sp_id] * len(hist)
                    if h % 2 == 0
                    else [next_sp_id] * len(hist)
                    for h, hist in enumerate(input_hists)
                ]
                assert len(input_hists) == len(token_type_ids)
                token_type_ids = (
                    [start_sp_id]
                    + list(chain.from_iterable(token_type_ids))
                    + [self.hparams.sp2_id]
                )
                assert len(token_type_ids) == len(input_ids)
                input_len = len(input_ids)

                input_ids = (
                    torch.LongTensor(input_ids).unsqueeze(0).to(self.hparams.device)
                )
                token_type_ids = (
                    torch.LongTensor(token_type_ids)
                    .unsqueeze(0)
                    .to(self.hparams.device)
                )

                output_ids = self._nucleus_sampling(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    input_len=input_len,
                )
                ### TODO: 왜 generate 안 쓴거야?
                # output_ids = self.model.generate(
                #     input_ids=input_ids, token_type_ids=token_type_ids, pad_token_id=self.args.eos_id,
                #     do_sample=True, top_p=self.args.top_p, max_length=self.args.max_len,
                #     output_hidden_states=True, output_scores=True, return_dict_in_generate=True,
                # ).sequences
                # output_ids = output_ids[0].tolist()[input_len:]
                res = self.hparams.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )

                print(f"Bot: {res}")
                input_hists.append(
                    [self.hparams.sp2_id] + self.hparams.tokenizer.encode(res)
                )

    def _nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        # Loop: input_len -> max_len
        # output -> logits -> prediction
        # sort -> cumsum -> idx_remove
        # sorted_probs에 idx_remove 적용 후 정규화
        # scatter_ => prob -> multinomial => idx
        # output_ids에 item 추가하고
        # Bot이 말한 idx도 input_ids에 붙이기
        for pos in range(input_len, self.hparams.max_len):
            # Q. 왜 자르는 거야? A. 마지막 logit 이용
            ### TODO: pos-1 == 마지막 idx랑 똑같지 않나? => pos-1 대신 -1 써도 될듯?
            logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][
                :, pos - 1
            ]
            prediction = F.softmax(logits, dim=-1)  # (1, V)

            sorted_probs, sorted_idxs = torch.sort(prediction, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            idx_remove = cumsum_probs > self.hparams.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            # idx_remove는 제거할 idx를 찾기
            sorted_probs[idx_remove] = 0.0
            # 정규화? 전체 합 = 1로 만듦
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)

            # Q. output.shape으로 만드는 이유가 있나? sorted_probs랑 output이랑 shape이 다른가?
            # A. 같은데 sorted_probs를 sorted_idxs를 이용해서 해당하는 index에 넣기 위해
            # multinomial
            # input: (m, cols) => output: (m, num_samples)
            probs = torch.zeros(logits.shape, device=self.hparams.device).scatter_(
                dim=-1, index=sorted_idxs, src=sorted_probs
            )
            idx = torch.multinomial(input=probs, num_samples=1)

            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)

            if idx_item == self.hparams.eos_id:
                break

            # bot이 말한 response도 input_ids에 붙이기
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.hparams.sp2_id]]).to(
                self.hparams.device
            )
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert token_type_ids.shape == input_ids.shape

        return output_ids


if __name__ == "__main__":
    # Program level arhparams   parser = argparse.ArgumentParser(description="Multi-turn chatbot baseed on GPT-2")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The random seed.")
    parser.add_argument(
        "--mode", type=str, required=True, help="The running mode: train or inference?"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="The name of the parent directory where data files are stored.",
    )
    parser.add_argument(
        "--train_prefix",
        type=str,
        default="train",
        help="The prefix of the train data files' name.",
    )
    parser.add_argument(
        "--valid_prefix",
        type=str,
        default="valid",
        help="The prefix of the validation data files' name.",
    )
    parser.add_argument(
        "--model_type", type=str, default="gpt2", help="The model type of GPT-2."
    )
    parser.add_argument("--bos_token", type=str, default="<bos>", help="The BOS token.")
    parser.add_argument(
        "--sp1_token", type=str, default="<sp1>", help="The speaker1 token."
    )
    parser.add_argument(
        "--sp2_token", type=str, default="<sp2>", help="The speaker2 token."
    )
    parser.add_argument("--gpu", type=int, default=0, help="The index of GPU to use.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of workers for data loading.",
    )
    # parser.add_argument(
    #     "--num_epochs", type=int, default=10, help="The number of total epochs."
    # )

    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="The maximum number of dialogue histories to include.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The top-p value for nucleus sampling decoding.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="saved_models",
        help="The directory name for saved checkpoints.",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="last",
        required=False,
        help="The name of the trained checkpoint. (without extension)",
    )
    parser.add_argument(
        "--end_command",
        type=str,
        default="Abort!",
        help="The command to stop the conversation when inferencing.",
    )
    parser = KoGPTTask.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logger.info(pformat(args))

    assert args.mode in ["train", "infer"]
    assert args.model_type in [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "skt/kogpt2-base-v2",
    ]

    args.data_dir = f"{args.data_dir}/{args.model_type}"
    args.ckpt_dir = f"{args.ckpt_dir}/{args.model_type}"

    # Tokenizer & Vocab
    logger.info("Loading the tokenizer...")
    args.tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    special_tokens = {
        "bos_token": args.bos_token,
        "additional_special_tokens": [args.sp1_token, args.sp2_token],
    }
    args.eos_token = args.tokenizer.eos_token
    num_new_tokens = args.tokenizer.add_special_tokens(special_tokens)

    vocab = args.tokenizer.get_vocab()
    args.vocab_size = len(vocab)
    args.bos_id = vocab[args.bos_token]
    args.eos_id = vocab[args.eos_token]
    args.sp1_id = vocab[args.sp1_token]
    args.sp2_id = vocab[args.sp2_token]

    if args.mode == "train":
        # best ckpt만 저장하기
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename="best_ckpt_{epoch:02d}-{val_loss:.2f}",
            verbose=True,
            save_last=True,
            monitor="val_loss",
            mode="min",
            save_top_k=-1,
        )

        dm = KoGPTDataModule(args)
        dm.setup("fit")
        args.num_batches = dm.num_batches
        model = KoGPTTask(args)
        model.train()

        if args.gpu == 1:
            trainer = pl.Trainer.from_argparse_args(
                args,
                callbacks=[checkpoint_callback],
            )
        else:
            trainer = pl.Trainer()
        trainer.fit(model=model, datamodule=dm)
        logger.info(f"best model path: {checkpoint_callback.best_model_path}")

    if args.mode == "infer":
        checkpoint = f"{args.ckpt_dir}/{args.ckpt_name}.ckpt"
        model = KoGPTTask.load_from_checkpoint(checkpoint)
        model.eval()
        model.infer()
