import json
from pathlib import Path
from collections import defaultdict
from pprint import pprint

import numpy as np
from tqdm import tqdm

from args import get_args
from loader.qa import DataSpecificsQA


'''
video/text split for movieqa
'''


def main():
    args = get_args(print_args=False)
    args.dataset = 'movieqa'
    assert args.dataset == 'movieqa'
    stats, num, total = run(args)
    print(f"{num}/{total}")

    res = {str(k): v for k, v in stats.items()}
    pprint(res)
    import ipdb; ipdb.set_trace()  # XXX DEBUG
    Path('./stats/').mkdir(exist_ok=True)
    with open('./stats/movieqa_split.json', 'w') as f:
        json.dump(res, f, indent=4)


def load_video_ids():
    dpath = Path('../../data/movieqa')
    path = dpath / 'frame_clip'
    ids = [p.stem.split('.')[0] for p in path.glob('*.video.pkl')]
    ids = set(ids)
    return ids


def run(args, use_tqdm: bool = True):
    dset = args.dataset
    data_path = f'../../data/{dset}'
    data_path = Path(data_path)
    get_specifics = DataSpecificsQA(args, data_path, dset)

    video_ids = load_video_ids()

    qname = 'qa'
    if args.use_only_hypo:
        qname = 'qa_hypo'

    hypo_path = data_path.parent / 'outputs' / args.dataset / qname / args.split / args.name
    '''
    if not hypo_path.is_dir():
        hypo_path = data_path.parent / 'outputs' / args.dataset / qname / args.name
    '''
    vid_path, valqa, gt_plots, get_vid_key, get_gt_index, unload_data = get_specifics()

    valqa = [unload_data(v) for v in valqa]
    valqa = {v[0]: v for v in valqa}
    data = {}
    for path in hypo_path.glob('*'):
        with open(path) as f:
            row = json.load(f)
            data[row['qid']] = row

    res = defaultdict(lambda: defaultdict(lambda: []))
    keys = list(valqa.keys())
    num = 0
    pbar = keys
    if use_tqdm:
        pbar = tqdm(keys)
    for qid in pbar:
        q_ = valqa[qid]
        if qid in data:
            num += 1
            hypo = data[qid]['hypos']
            tgt = q_[-2]
            vid = data[qid]['vis_key']
            is_video = vid in video_ids
            if tgt is not None:
                for k, v in hypo.items():
                    lh = [v['likelihood'][str(i)] for i in range(5)]
                    decision = np.argmax(lh)
                    correct = int(decision) == tgt
                    if is_video:
                        res['video'][k].append(correct)
                    else:
                        res['subtitle'][k].append(correct)
                    res['all'][k].append(correct)

    stats = {k_: {k: (np.mean(v), len(v)) for k, v in v_.items()}
             for k_, v_ in res.items()}
    return stats, num, len(keys)


if __name__ == '__main__':
    main()
