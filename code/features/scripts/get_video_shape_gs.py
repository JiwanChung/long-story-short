import os
import re
import math
import shutil
import json
import pickle
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import concurrent.futures

from tqdm import tqdm
from simple_parsing import ArgumentParser
import jax
import numpy as np
import skvideo.io
from PIL import Image
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
import gswrap


@dataclass
class Config:
    bucket_name: str = 'gs://temp'
    blob_dir: str = 'movieqa/video_clips'
    out_blob_dir: str = 'movieqa/meta'
    chunk_size: int = 200

    clip_model_type: str = 'ViT-B/32'
    data_path: str = './data/video_clips'
    out_dir: str = './data/meta'
    margin: int = 10
    num_frames: int = 10


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
        filenames = [p for p in filenames if p.endswith('.video.mp4')]
        return filenames


    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    filenames = gs_get_filenames(args, client, bucket)
    chunks = list(chunk_list(filenames, args.chunk_size))


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
    out_dir = Path(args.out_dir)


    '''
    paths = list(Path(args.data_path).glob('*.video.mp4'))
    '''

    pbar = tqdm(filenames, desc='clips')
    shapes = defaultdict(lambda: [])
    all_keys = set()
    for chunk in chunks:
        if Path(args.data_path).is_dir():
            shutil.rmtree(args.data_path)
        Path(args.data_path).mkdir(parents=True)
        download_blobs(args, bucket, chunk, args.data_path)
        for filename in chunk:
            key = Path(filename).parent.name
            all_keys.add(key)
            pbar.update(1)
            path = Path(args.data_path) / filename
            meta = skvideo.io.ffprobe(path)
            if 'video' in meta:
                x = meta['video']
                shapes[key].append((x['@height'], x['@width']))


    shapes_ = {}
    for k, v in shapes.items():
        if all([v2 == v[0] for v2 in v]):
            shapes_[k] = v[0]
        else:
            print(f'mismatch: {k}, {v}')

    for k in list(all_keys):
        if k not in shapes_:
            print(f'key not found: {k}')

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'shapes.pkl', 'wb') as f:
        pickle.dump(shapes_, f)


    files = list(out_dir.glob('*.pkl'))
    print('uploading')
    for local_file in tqdm(files, total=len(files), desc='upload'):
        name = local_file.name
        blob = bucket.blob(f'{args.out_blob_dir}/{name}')
        blob.upload_from_filename(str(local_file))


    print('done')


if __name__ == '__main__':
    main()
