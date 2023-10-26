import json
from pathlib import Path
from collections import defaultdict
from pprint import pprint

import numpy as np
from tqdm import tqdm

from args import get_args
from loader.qa import DataSpecificsQA
from utils import get_bent, get_bmask, softmax


def main():
    args = get_args(print_args=False)
    stats, num, total = run(args)
    print(f"{num}/{total}")
    pprint(stats)


def mix_lh(args, txt, vis):
    lhs = np.array(txt)
    vis = np.array(vis)
    vis = softmax(vis, temp=args.vis_temperature)

    score = lhs
    ent = get_bent(lhs)
    ent_vis = get_bent(vis)
    if ent > 0.4:
        score = vis * lhs
    return score


    '''
    lhs = lhs / lhs.sum(-1, keepdims=True)
    lhs = softmax(np.log(lhs), temp=lhs_temp)
    '''
    '''
    vis = np.array(vis)
    bent_vis = softmax(vis, temp=4e-3)
    vis = softmax(vis, temp=args.vis_temperature)

    score = txt
    ent_txt = get_bent(txt)
    ent_vis = get_bent(bent_vis)
    ## DEBUG
    return vis
    # if ent_txt >= 0.5 and ent_vis >= 0.5:
    if ent_txt >= 0.5:
        score = vis * txt
    return score
    '''


def run(args, use_tqdm: bool = True):
    dset = args.dataset
    data_path = f'../../data/{dset}'
    data_path = Path(data_path)
    get_specifics = DataSpecificsQA(args, data_path, dset)

    vis_check_name = 'vis_check'
    # vis_check_name = 'vis_check_rand'

    qname = 'qa'
    if args.use_only_hypo:
        qname = 'qa_hypo'

    hypo_path = data_path.parent / 'outputs' / args.dataset / qname / args.split / args.name
    if not hypo_path.is_dir():
        hypo_path = data_path.parent / 'outputs' / args.dataset / qname / args.name
    vid_path, qa, gt_plots, get_vid_key, get_gt_index, unload_data = get_specifics()

    vis_check_path = data_path.parent / 'outputs' / args.dataset / vis_check_name / args.split / args.name
    vis_check = {p.stem: json.load(open(p)) for p in vis_check_path.glob('*.json')}

    if args.dataset == 'movieqa':
        # get video split
        video_split = set([v['qid'] for v in qa if len(v['video_clips']) > 0])
    qa = [unload_data(v) for v in qa]
    qa = {v[0]: v for v in qa}
    data = {}
    for path in hypo_path.glob('*'):
        with open(path) as f:
            row = json.load(f)
            data[row['qid']] = row

    res = defaultdict(lambda: [])
    keys = list(qa.keys())
    num = 0
    pbar = keys
    if use_tqdm:
        pbar = tqdm(keys)
    for qid in pbar:
        q_ = qa[qid]
        if qid in data:
            hypo = data[qid]['hypos']
            tgt = q_[-2]

            if str(qid) not in vis_check:
                continue
            if args.dataset == 'movieqa' and qid not in video_split:
                continue

            num += 1
            vis = vis_check[str(qid)]
            vis = {int(k): v for k, v in vis.items()}
            vis_keys = sorted([k for k in vis.keys()])
            vis_arr = np.stack([vis[k] for k in vis_keys], axis=0)

            def get_correct(lh):
                decision = np.argmax(lh)
                correct = int(decision) == tgt
                return correct

            for k, v in hypo.items():
                lh = [v['likelihood'][str(i)] for i in range(5)]
                lh = np.array(lh)
                res[f'{k}-text'].append(get_correct(lh))

                dec_mean = get_correct(mix_lh(args, lh, vis_arr.mean(0)))
                res[f'{k}-vis/max'].append(get_correct(mix_lh(args, lh, vis_arr.max(0))))
                res[f'{k}-vis/mean'].append(dec_mean)
                # if 'lookup' in v and 'index' in v['lookup']:
                if not k.endswith('/none'):
                    ids = v['lookup']['index']
                    if ids is not None:
                        if isinstance(ids, int):
                            ids = [ids]
                        ids = [v2 for v2 in ids if v2 is not None and v2 < vis_arr.shape[0]]
                        if len(ids) > 0:
                            vis_sub_arr = [vis_arr[idx] for idx in ids if idx in vis_arr]
                            if len(vis_sub_arr) > 0:
                                vis_sub_arr = np.stack(vis_sub_arr, axis=0)
                                res[f'{k}-vis/index'].append(get_correct(mix_lh(args, lh, vis_sub_arr.mean(0))))
                            else:
                                res[f'{k}-vis/index'].append(dec_mean)
                        else:
                            res[f'{k}-vis/index'].append(dec_mean)
                    else:
                        res[f'{k}-vis/index'].append(dec_mean)

                '''
                bent_v = 0.5
                if get_bent(lh) >= bent_v:
                    res[f'{k}-text/b>={bent_v}'].append(get_correct(lh))
                    res[f'{k}-vis/mean/b>={bent_v}'].append(dec_mean)
                    if not k.endswith('/none'):
                        res[f'{k}-vis/index/b>={bent_v}'].append(res[f'{k}-vis/index'][-1])
                else:
                    res[f'{k}-text/b<{bent_v}'].append(get_correct(lh))
                    res[f'{k}-vis/mean/b<{bent_v}'].append(dec_mean)
                    if not k.endswith('/none'):
                        res[f'{k}-vis/index/b<{bent_v}'].append(res[f'{k}-vis/index'][-1])
                '''

    stats = {k: (np.mean(v), len(v)) for k, v in res.items()}
    return stats, num, len(keys)


if __name__ == '__main__':
    main()
