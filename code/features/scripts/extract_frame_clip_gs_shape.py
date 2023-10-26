import os
import re
import math
import shutil
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
import concurrent.futures

from tqdm import tqdm
from simple_parsing import ArgumentParser
import jax
import clip_jax
import numpy as np
# import skvideo.io
import torchvision.io
from PIL import Image
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
import gswrap


@dataclass
class Config:
    bucket_name: str = 'gs://temp'
    blob_dir: str = 'movieqa/video_clips'
    shape_blob_path: str = 'movieqa/meta/shapes.pkl'
    out_blob_dir: str = 'movieqa/frame_clip'
    chunk_size: int = 20

    clip_model_type: str = 'ViT-B/32'
    data_path: str = './data/video_clips'
    shape_path: str = './data'
    out_dir: str = './data/frame_clip'
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


    '''
    if Path(args.shape_path).is_dir():
        shutil.rmtree(args.shape_path)
    '''
    Path(args.shape_path).mkdir(parents=True, exist_ok=True)
    sb_path = Path(args.shape_blob_path)
    shapes = [sb_path.name]
    download_blobs(args, bucket, shapes, args.shape_path, sb_path.parent)
    shape_path = Path(args.shape_path) / sb_path.name
    with open(shape_path, 'rb') as f:
        shapes = pickle.load(f)


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

    def check_valid(path):
        name = path.name
        name = name.split('.')
        st = int(name[1][3:])
        et = int(name[2][3:])
        return et - st >= 10


    '''
    def load_video(path, meta):
        return skvideo.io.vread(str(path), **meta)
    '''


    def load_video(path, meta):
        visual, audio, info = torchvision.io.read_video(str(path))
        assert visual is not None
        data = visual.numpy()
        return data


    pbar = tqdm(filenames, desc='clips')
    zero_lengths = []
    errors = []
    emptry_frames = []

    for chunk in chunks:
        if Path(args.data_path).is_dir():
            shutil.rmtree(args.data_path)
        Path(args.data_path).mkdir(parents=True)
        download_blobs(args, bucket, chunk, args.data_path)
        for filename in chunk:
            key = Path(filename).name.split('.')[0]
            meta = shapes[key]
            meta = {'height': int(meta[0]), 'width': int(meta[1])}

            pbar.update(1)
            path = Path(args.data_path) / filename
            out_path = out_dir / f'{path.stem}.pkl'
            if out_path.is_file():
                print(f'skipping {path.stem}')
                continue

            if not check_valid(path):
                print(f'invalid video {path.stem}')
                continue

            out_dir.mkdir(exist_ok=True, parents=True)
            try:
                data = load_video(path, meta)
            except Exception as e:
                print(key, meta)
                print(e)
                print(f'errorneous video {path.stem}')
                continue

            if len(data) < 1:
                print(f'zero length video {path.stem}')
                zero_lengths.append(path.stem)
                continue
            elif len(data) == 1:
                frs_raw = data
                indices = np.array([0])
            else:
                if len(data) - 2 * args.margin - 1 >= args.num_frames:
                    indices = np.linspace(args.margin, len(data) - args.margin, args.num_frames)
                else:
                    indices = np.linspace(0, len(data) - 1, min(len(data) - 1, args.num_frames))
                indices = indices.astype(int)
                frs_raw = [data[i] for i in indices]

            if len(frs_raw) == 0:
                print(f'zero length video {path.stem}')
                zero_lengths.append(path.stem)
                continue
            frs = np.stack([jax_preprocess(Image.fromarray(image)) for image in frs_raw], axis=0)  # b c
            if (frs.std(-1).mean(-1).mean(-1) == 0).all():
                print(f'empty_frames {path.stem}')
                empty_frames.append(path.stem)
                continue

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

            res = {'timestamps': indices, 'features': feats, 'frames': frs_raw}
            with open(out_path, 'wb') as f:
                pickle.dump(res, f)


    files = list(out_dir.glob('*.pkl'))
    print('uploading')
    for local_file in tqdm(files, total=len(files), desc='upload'):
        name = local_file.name
        blob = bucket.blob(f'{args.out_blob_dir}/{name}')
        blob.upload_from_filename(str(local_file))

    print('done')


if __name__ == '__main__':
    main()
