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

    @classmethod
    def from_toml(cls, file: str) -> "Config":
        with open(file, "rb") as f:
            config = tomllib.load(f)

        data_path = config["data_path"]
        if data_path is None:
            raise ValueError("Must provide data_path")

        training_iter = config.get("train_iter", 5000)
        batch_size = config.get("batch_size", 32)
        context_len = config.get("context_len", 256)
        d_model = config.get("d_model", 512)
        d_ff = config.get("d_ff", 1344)
        num_heads = config.get("num_heads", 16)
        num_layers = config.get("num_layers", 4)
        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 0.9)
        max_lr = config.get("max_lr", 1e2)
        min_lr = config.get("min_lr", 1e-2)
        warmup_iters = config.get("warmup_iters", 100)
        cosine_cycle_iters = config.get("cosine_cycle_iters", 3500)
        checkpoint_dir = config["checkpoint_dir"]

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
        )
