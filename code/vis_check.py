import os
import json
import math
import re
import random
from pathlib import Path
from itertools import chain, product
from collections import defaultdict

from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp
import clip_jax

from args import get_args
from loader.base import LoadSubtt
from loader.vis import DataSpecificsVis
from utils import get_chunks
from tokenizer import tokenize as clip_tokenize


def main():
    args = get_args()
    dset = args.dataset
    data_path = f'../../data/{dset}'
    data_path = Path(data_path)
    get_specifics = DataSpecificsVis(args, data_path, dset)
    load_subtt = LoadSubtt(args, data_path, dset)

    extractor = Extractor(args)

    run(args, extractor, data_path, get_specifics, load_subtt)


class Extractor:
    def __init__(self, args):
        self.devices = jax.local_devices()
        print(f"jax devices: {self.devices}")
        print("loading model")
        image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load(args.clip_model_type, "cpu")
        self.jax_params = jax.device_put_replicated(jax_params, self.devices)
        self.text_fn = jax.pmap(text_fn)
        print("loaded model")

    def __call__(self, frs):
        feats = jnp.array(self.text_fn(self.jax_params, frs))
        return feats


def run_with_padding(extractor, txts):
    frs = clip_tokenize(txts, truncate=True)

    # run with padding
    batch_size = frs.shape[0]
    devices = extractor.devices
    if frs.shape[0] % len(devices) != 0:
        div = math.ceil(frs.shape[0] / len(devices))
        diff = div * len(devices) - frs.shape[0]
        padder = np.repeat(frs[:1], diff, axis=0)
        frs = np.concatenate([frs, padder], axis=0)
    frs = frs.reshape(len(devices), -1, *frs.shape[1:])
    feats = extractor(frs)
    feats = feats.reshape(-1, feats.shape[-1])
    feats = feats[:batch_size]
    return feats


def normalize(v, eps=1e-16):
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


def expand(x, length=None):
    if length is None:
        length = max([v.shape[0] for v in x])
    ids = np.linspace(0, x.shape[0] - 1, length)
    ids = np.round(ids).astype(int)
    x2 = np.stack([x[i] for i in ids], axis=0)
    return x2


def get_med_frame(v):
    med = len(v) // 2
    return v[med][:, None]


def run_model(extractor, vis, txts):
    vis = [expand(v) for v in vis]
    # vis = [get_med_frame(v) for v in vis]

    vis = np.stack(vis, axis=0)
    vis = normalize(vis)

    txt_feats = run_with_padding(extractor, txts)  # N D
    txt_feats = normalize(txt_feats)
    if len(vis.shape) == 3:
        vis = vis.mean(1)
    sim = jnp.einsum('vd,nd->nv', vis, txt_feats)
    return sim


def load_hypo(path):
    paths = {p.stem: p for p in path.glob('*.json')}
    res = {}
    for k, path in paths.items():
        with open(path) as f:
            res[k] = json.load(f)
    return res


def run(args, extractor, data_path, get_specifics, load_subtt):
    out_dir = data_path.parent / 'outputs' / args.dataset / 'vis_check' / args.split / args.name
    out_dir.mkdir(exist_ok=True, parents=True)
    vid_path, qa, gt_plots, get_vid_key, get_gt_index, unload_data, load_vis = get_specifics()

    batches = list(get_chunks(qa, args.batch_size))
    pbar = tqdm(batches)
    num_vid = 0
    for batch in pbar:
        b_vis = []
        b_txt = []
        b_vis_keys = []
        b_txt_keys = []
        qs = []
        id_map_vis = defaultdict(lambda: [])
        id_map_txt = defaultdict(lambda: [])
        for q_ in batch:
            qid, question, answers, answer, key = unload_data(q_)
            txt = [f"{question.strip()} {answer.strip()}" for answer in answers]
            txt = answers
            vis_ = load_vis(key)
            if vis_ is not None:
                qs.append(q_)
                vis, vis_keys = vis_
                txt_keys = [(qid, answer) for answer in txt]

                pre_len = len(b_vis_keys)
                b_vis_keys.extend(vis_keys)
                post_len = len(b_vis_keys)
                ids = list(range(pre_len, post_len))
                id_map_vis[qid].extend(ids)

                pre_len = len(b_txt_keys)
                b_txt_keys.extend(txt_keys)
                post_len = len(b_txt_keys)
                ids = list(range(pre_len, post_len))
                id_map_txt[qid].extend(ids)

                b_vis.extend(vis)
                b_txt.extend(txt)

        if len(b_vis) == 0:
            continue
        sim = run_model(extractor, b_vis, b_txt)  # txt x vis
        b_vis_keys = np.array(b_vis_keys)

        for q_ in qs:
            qid, question, answers, answer, key = unload_data(q_)

            ids_txt = np.array(id_map_txt[qid])
            ids_vis = np.array(id_map_vis[qid])
            sim_ = sim[ids_txt][:, ids_vis]

            vis_keys_ = b_vis_keys[ids_vis]
            sim_ = sim_.transpose()  # vis x txt

            res = defaultdict(lambda: [])
            for vkey, scores in zip(vis_keys_, sim_):
                res[vkey].append(np.array(scores))
            res = {int(k): np.mean(v, axis=0).tolist() for k, v in res.items()}

            out_path = out_dir / f"{qid}.json"
            with open(out_path, 'w') as f:
                json.dump(res, f, indent=4)
            num_vid += 1
        pbar.set_description(f'num_vid: {num_vid}')


if __name__ == '__main__':
    main()
