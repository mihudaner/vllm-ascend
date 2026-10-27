"""Create a tiny independent fixture and exercise the file-generation interface."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from test_golden import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite fixture')
    value = fixture()
    value['config'] = asdict(value.pop('cfg'))
    torch.save(value, args.output)


if __name__ == '__main__':
    main()
