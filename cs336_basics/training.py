import logging
import pathlib
import time
from argparse import ArgumentParser

import numpy as np

from cs336_basics.adamw import AdamW, get_lr_cosine_schedule
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.config import Config
from cs336_basics.dataloader import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.transformer import TransformerLM

parser = ArgumentParser(description="Training script config parser")
parser.add_argument("--config", type=str, help="model training config")

logging.basicConfig(handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/{time.time()}.log")])


def main(config_file: str):
    cfg = Config.from_toml(config_file)
    logging.info(f"starting new experiment, config: {cfg}")

    if cfg.data_path is None:
        raise ValueError("must provide data path")

    dataset = np.memmap(cfg.data_path, float)
    model = TransformerLM(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.vocab_size, cfg.context_len, cfg.num_layers)
    opt = AdamW({}, cfg.lr, cfg.weight_decay, cfg.betas, cfg.eps)

    for train_it in range(cfg.training_iter):
        iter_lr = get_lr_cosine_schedule(train_it, cfg.max_lr, cfg.min_lr, cfg.warmup_iters, cfg.cosine_cycle_iters)
        opt.update_lr(iter_lr)
        opt.zero_grad()
        total_loss = 0
        for data, target in get_batch(dataset, cfg.batch_size, cfg.context_len, "mps"):
            logits = model.forward(data)
            loss = cross_entropy(logits, target)
            total_loss += loss
            loss.backward()
            opt.step()

        logging.info(f"iteration {train_it}, total loss {total_loss}")

        checkpoint_path = pathlib.Path(cfg.checkpoint_dir) / f"iter_{train_it}"
        save_checkpoint(model, opt, train_it, checkpoint_path)
        logging.info(f"saved checkpoint to {checkpoint_path} for training iter {train_it}")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config is None:
        raise ValueError("Must pass in training config as a toml file")

    main(args.config)
