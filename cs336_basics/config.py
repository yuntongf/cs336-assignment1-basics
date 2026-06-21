import tomllib
from dataclasses import dataclass


@dataclass
class Config:
    data_path: str
    training_iter: int
    batch_size: int
    context_len: int
    d_model: int
    d_ff: int
    num_heads: int
    num_layers: int
    lr: float
    weight_decay: float
    max_lr: float
    min_lr: float
    warmup_iters: int
    cosine_cycle_iters: int
    checkpoint_dir: str
    betas: tuple[float, float]
    eps: float
    vocab_size: int
    log_dir: str

    @classmethod
    def from_toml(cls, file: str) -> "Config":
        with open(file, "rb") as f:
            config = tomllib.load(f)

        data_path = config["data_path"]
        if data_path is None:
            raise ValueError("Must provide data_path")

        training_iter = config["train_it"] or 1000
        batch_size = config["batch_size"] or 1024
        context_len = config["context_len"] or 2048
        d_model = config["d_model"] or 1024
        d_ff = config["d_ff"] or 512
        num_heads = config["num_heads"] or 4
        num_layers = config["num_layers"] or 8
        lr = config["lr"] or 1e1
        weight_decay = config["weight_decay"] or 0.9
        max_lr = config["max_lr"] or 1e3
        min_lr = config["min_lr"] or 1e1
        warmup_iters = config["warmup_iters"] or 5
        cosine_cycle_iters = config["cosine_cycle_iters"] or 100
        checkpoint_dir = config["checkpoint_dir"]
        log_dir = config["log_dir"]

        betas = (0.9, 0.95)
        eps = 1e-6
        vocab_size = 10_000

        return cls(
            data_path,
            training_iter,
            batch_size,
            context_len,
            d_model,
            d_ff,
            num_heads,
            num_layers,
            lr,
            weight_decay,
            max_lr,
            min_lr,
            warmup_iters,
            cosine_cycle_iters,
            checkpoint_dir,
            betas,
            eps,
            vocab_size,
            log_dir,
        )
