import argparse

def build_parser():
    p = argparse.ArgumentParser(description="ElderGuard preprocessing and model comparison")
    p.add_argument("--db-path", type=str, required=False, default=None)
    p.add_argument("--table", type=str, required=False, default=None)
    return p
