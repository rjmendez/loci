"""Fixture: KEY-LIKE literal shapes — os.environ.get/subscript and
dict.get/subscript, both read and write positions."""
import os


def read_port():
    return os.environ.get("LOCI_PORT")


def set_port(v):
    os.environ["LOCI_PORT"] = v


def read_config_key(cfg: dict):
    return cfg.get("cfg_lookup_only")   # consumer-only key: nothing ever sets it


def write_config_key(cfg: dict, v):
    cfg["cfg_write_only"] = v           # producer-only key: nothing ever reads it
