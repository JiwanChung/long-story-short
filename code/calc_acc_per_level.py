import json
from pathlib import Path
from collections import defaultdict
from pprint import pprint

import numpy as np
from tqdm import tqdm

from args import get_args
from loader.qa import DataSpecificsQA


'''
Per Level accuracy for DramaQA
'''


def main():
    args = get_args(print_args=False)
    args.dataset = 'dramaqa'
    assert args.dataset == 'dramaqa'
    dset = args.dataset
    data_path = f'../../data/{dset}'
    data_path = Path(data_path)
    get_specifics = DataSpecificsQA(args, data_path, dset)
    stats, num, total = run(args, data_path, get_specifics)
    print(f"{num}/{total}")

    res = {str(k): v for k, v in stats.items()}
    Path('./stats/').mkdir(exist_ok=True)
    with open('./stats/per_level.json', 'w') as f:
        json.dump(res, f, indent=4)
    pprint(res)


def run(args, data_path, get_specifics):
    qname = 'qa'
    if args.use_only_hypo:
        qname = 'qa_hypo'
    hypo_path = data_path.parent / 'outputs' / args.dataset / qname / args.split / args.name

    vid_path, valqa, gt_plots, get_vid_key, get_gt_index, unload_data = get_specifics()

    valqa = {v['qid']: v for v in valqa}
    data = {}
    for path in hypo_path.glob('*'):
        with open(path) as f:
            row = json.load(f)
            data[row['qid']] = row

    res = defaultdict(lambda: defaultdict(lambda: []))
    keys = list(valqa.keys())
    num = 0
    for qid in tqdm(keys):
        q_ = valqa[qid]
        if qid in data:
            num += 1
            hypo = data[qid]['hypos']
            tgt = q_['correct_idx']
            level = q_['q_level_logic']
            for k, v in hypo.items():
                lh = [v['likelihood'][str(i)] for i in range(5)]
                decision = np.argmax(lh)
                correct = int(decision) == tgt
                res[k][level].append(correct)
                res[k]['total'].append(correct)

    stats = {k2: {k: (np.mean(v), len(v)) for k, v in v2.items()}
             for k2, v2 in res.items()}
    return stats, num, len(keys)


if __name__ == '__main__':
    main()
