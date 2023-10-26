import os
from typing import Optional
from dataclasses import dataclass, field

from simple_parsing import ArgumentParser
import yaml

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@dataclass
class Params:
    config: Optional[str] = None
    debug: bool = False

    name: str = 'base'
    out_name: Optional[str] = None
    dataset: str = 'movieqa'
    split: str = 'val'
    exp_keys: Optional[str] = None

    max_plot_tokens: int = 256
    max_char_tokens: int = 512
    max_qa_tokens: int = 100

    prev_prompt_num: int = 0

    # for chunking movieqa
    chunking_method: str = 'base'  #
    chunk_size: int = 100

    # for visual checking on GPU
    batch_size: int = 64

    # visual checking
    vis_temperature: float = 2e-2

    use_caption: bool = False  # NotImplemented
    use_only_hypo: bool = False
    do_mp: bool = False  # multiprocess
    num_workers: int = 40
    clip_model_type: str = 'ViT-B/32'

    engine: str = "text-davinci-002"


def get_args(Config=Params, print_args: bool = True):
    parser = ArgumentParser()
    parser.add_arguments(Config, dest='config')
    args = parser.parse_args()
    args = args.config
    with open('../secrets.yml') as f:
        config = yaml.safe_load(f)
    args.api_key = config['api_key']

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)
    if args.out_name is None:
        args.out_name = args.name
    if print_args:
        print(args)
    return args
