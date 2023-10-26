from itertools import chain
from collections import defaultdict
from pathlib import Path

from .srt import load_srt


def load_subtt_movieqa(args, boundaries, key, subt_path, matidx_path):
    if key in boundaries:
        boundary = boundaries[key]
        sub_list, chunks, _, _ = load_subtt_with_boundary(subt_path, boundary, matidx_path)
    else:
        sub_list = load_srt(subt_path)
        sub_list = [v.content for v in sub_list if len(v.content) > 0]
        chunks = list(get_chunks(sub_list, args.chunk_size, args.chunking_method))
    return chunks


def get_chunks(li, size, method=''):
    if method == 'windowed':
        step_size = size // 2
        return list(get_windowed(li, size, step_size))
    else:
        return [li[i:i + size] for i in range(0, len(li), size)]


def get_windowed(li, size, step_size):
    i = 0
    while True:
        yield li[i: i + size]
        i += step_size
        if i >= len(li):
            break


def load_scene_boundaries(path):
    res = defaultdict(lambda: [])
    with open(path) as f:
        for line in f:
            p = Path(line.strip())
            name = p.name
            if name.endswith('video.mp4'):
                name = name.split('.')
                key, st, et = name[:3]
                st = int(st.split('-')[1])
                et = int(et.split('-')[1])
                res[key].append((st, et))
    res = dict(res)
    return res


def load_scene_boundary(x):
    return sorted(x, key=lambda v: v[0])


def get_chunk_idx(btimemap, sub_et):
    for i, tm in enumerate(btimemap):
        if tm >= sub_et:
            return i
    return len(btimemap) - 1


def load_subtt_with_boundary(subt_path, boundary, matidx_path):
    boundaries = load_scene_boundary(boundary)
    subt = load_srt(subt_path)
    key = subt_path.stem
    matidx_path = matidx_path / f'{key}.matidx'
    timemap = {}
    with open(matidx_path) as f:
        for line in f:
            fr, ts = line.strip().split()
            timemap[int(fr)] = float(ts)

    btimemap = [timemap[v[1]] for v in boundaries]
    curr = 0
    chunks = defaultdict(lambda: [])
    for sub in subt:
        if sub.content:
            sub_et = sub.end.total_seconds()
            i = get_chunk_idx(btimemap, sub_et)
            chunks[i].append(sub.content)
    chunks = dict(chunks)
    keys = sorted(list(chunks.keys()))
    keys = [k for k in keys if chunks[k]]
    chunks_ = [chunks[k] for k in keys]
    sub_list = list(chain(*chunks_))
    ets = [btimemap[k] if k in btimemap else None for k in keys]
    bbs = [boundaries[k] for k in keys]
    return sub_list, chunks_, ets, bbs
