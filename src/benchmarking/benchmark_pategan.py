"""Train and generate synthetic stroke data with PATE-GAN (synthcity)."""

import _bootstrap  # noqa: F401

from benchmarking.train import train_and_generate

if __name__ == "__main__":
    train_and_generate("pategan")
