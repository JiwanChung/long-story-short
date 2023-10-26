import os
import re
import math
import shutil
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
import concurrent.futures
from collections import defaultdict
from itertools import chain

from tqdm import tqdm
from simple_parsing import ArgumentParser
import jax
import clip_jax
import numpy as np
from PIL import Image
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
import gswrap


@dataclass
class Config:
    bucket_name: str = 'gs://temp'
    blob_dir: str = 'PororoQA/Scenes_Dialogues'
    out_blob_dir: str = 'PororoQA/frame_clip'
    chunk_size: int = 20
    depth_dir: int = 3
    filetype: bool = '.jpg'

    clip_model_type: str = 'ViT-B/32'
    data_path: str = './data/frames'
    out_dir: str = './data/frame_clip'


def get_args():
    parser = ArgumentParser()
    parser.add_arguments(Config, dest='config')
    args = parser.parse_args()
    args = args.config
    return args


def run(_args):
    args = Config()
    for k, v in dir(_args):
        if hasattr(k, args):
            setattr(args, k, v)
    _run(args)


def main():
    args = get_args()
    _run(args)


def _run(args):

    def get_client(args):
        client = storage.Client()
        gswrap_client = gswrap.Client()
        bucket = Bucket.from_string(args.bucket_name, client=client)
        return client, bucket


    client, bucket = get_client(args)


    def gs_get_filenames(args, client, bucket):
        paths = client.list_blobs(bucket, prefix=args.blob_dir, max_results=100000000)
        filenames = []
        for path in tqdm(paths):
            name = path.name[len(args.blob_dir) + 1:]
            filenames.append(name)
        filenames = [p for p in filenames if p.endswith(args.filetype) and '.ipynb' not in p]
        return filenames


    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    filenames = gs_get_filenames(args, client, bucket)
    res = defaultdict(lambda: [])
    for path in filenames:
        path = Path(path)
        res[str(path.parent)].append(path)
    filenames = res
    fname_keys = list(filenames.keys())

    chunks = list(chunk_list(fname_keys, args.chunk_size))


    def multi_blob(bucket, blob_list, max_workers = None):
        def process(bucket, fpath, local_vid):
            blob = bucket.blob(fpath)
            if blob.exists():
                blob.download_to_filename(local_vid)
            else:
                tqdm.write(f'file not found in bucket: {blob.name}')
            return

        # None is ThreadPoolExecutor max_workers default. 1 is single-threaded.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                    executor.submit(
                        process,
                        bucket,
                        fpath,
                        local_vid)
                    for fpath, local_vid in blob_list
                    ]
            for future in tqdm(futures, total=len(blob_list), desc='download'):
                _ = future.result()
        return True


    def download_blobs(args, bucket, chunk, path, blob_dir=None):
        if blob_dir is None:
            blob_dir = args.blob_dir

        blob_list = []
        path = Path(path)
        for filename in chunk:
            fname = str(Path(blob_dir) / filename)
            local_vid = str(path / filename)
            if not os.path.exists(local_vid):
                os.makedirs(str(Path(local_vid).parent), exist_ok=True)
                blob_list.append((fname, local_vid))
        multi_blob(bucket, blob_list)


    print("loading")
    devices = jax.local_devices()
    print(f"jax devices: {devices}")
    image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load(args.clip_model_type, "cpu")
    jax_params = jax.device_put_replicated(jax_params, devices)
    image_fn = jax.pmap(image_fn)
    out_dir = Path(args.out_dir)


    '''
    paths = list(Path(args.data_path).glob('*.video.mp4'))
    '''

    pbar = tqdm(filenames, desc='clips')
    for chunk in chunks:
        if Path(args.data_path).is_dir():
            shutil.rmtree(args.data_path)
        Path(args.data_path).mkdir(parents=True)
        fnames_sub = [filenames[key] for key in chunk]
        fnames_sub = list(chain(*fnames_sub))
        download_blobs(args, bucket, fnames_sub, args.data_path)

        for fname_key in chunk:
            paths = filenames[fname_key]
            paths = [Path(args.data_path) / '/'.join(p.parts[-args.depth_dir:]) for p in paths]
            img_keys = [p.stem for p in paths]
            key = fname_key.replace('/', '_')
            pbar.update(1)
            out_path = out_dir / f'{key}.pkl'
            if out_path.is_file():
                print(f'skipping {key}')
                continue
            out_dir.mkdir(exist_ok=True, parents=True)
            frs = np.stack([jax_preprocess(Image.open(image).convert('RGB')) for image in paths], axis=0)  # b c

            # run with padding
            batch_size = frs.shape[0]
            if frs.shape[0] % len(devices) != 0:
                div = math.ceil(frs.shape[0] / len(devices))
                diff = div * len(devices) - frs.shape[0]
                padder = np.repeat(frs[:1], diff, axis=0)
                frs = np.concatenate([frs, padder], axis=0)
            frs = frs.reshape(len(devices), -1, *frs.shape[1:])
            feats = np.array(image_fn(jax_params, frs))
            feats = feats.reshape(-1, feats.shape[-1])
            feats = feats[:batch_size]

            res = dict(zip(img_keys, feats))

            with open(out_path, 'wb') as f:
                pickle.dump(res, f)


    files = list(out_dir.glob('*.pkl'))
    assert len(filenames) == len(files)
    print('uploading')
    for local_file in tqdm(files, total=len(files), desc='upload'):
        name = local_file.name
        blob = bucket.blob(f'{args.out_blob_dir}/{name}')
        blob.upload_from_filename(str(local_file))

    print('done')


if __name__ == '__main__':
    main()
