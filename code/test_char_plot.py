from pathlib import Path
from pprint import pprint

from args import get_args
from loader.base import DataSpecifics, LoadSubtt
from get_char_plot import run as _run


def main():
    args = get_args()
    args.debug = True
    res = {}
    for dataset in ['pororoqa', 'dramaqa', 'movieqa', 'tvqa']:
        print(dataset)
        _res = run(args, dataset)
        res[dataset] = _res
    pprint(res)
    import ipdb; ipdb.set_trace()  # XXX DEBUG


def run(args, dataset):
    args.dataset = dataset
    dset = args.dataset
    data_path = f'../../data/{dset}'
    data_path = Path(data_path)
    get_specifics = DataSpecifics(args, data_path, dset)
    load_subtt = LoadSubtt(args, data_path, dset)
    res = _run(args, data_path, get_specifics, load_subtt)
    return res


if __name__ == '__main__':
    main()
