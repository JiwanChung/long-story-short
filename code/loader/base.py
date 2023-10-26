import json
from pathlib import Path

from .srt import load_srt
from .boundary import load_scene_boundaries, load_subtt_movieqa, get_chunks
from .dramaqa import get_keys_dramaqa, get_inits_dramaqa
from .starters import Starters


class DataSpecificsBase:
    def __init__(self, args, data_path, dataset: str = 'tvqa'):
        self.args = args
        self.split = args.split
        self.data_path = data_path
        self.dataset = dataset
        self.run = getattr(self, f"run_{self.dataset}")
        if hasattr(self, f'init_{self.dataset}'):
            getattr(self, f"init_{self.dataset}")()

    def __call__(self):
        vid_path, vids, get_starter, prompts = self.run()
        vids = sorted(vids)
        return vid_path, vids, get_starter, prompts


class DataSpecifics(DataSpecificsBase):
    def run_tvqa(self):
        vid_path = self.data_path / 'tvqa_subtitles'
        vids = load_vids_tvqa(self.data_path, vid_path)
        get_starter = Starters('tvqa')
        prompts = {}
        return vid_path, vids, get_starter, prompts

    def run_pororoqa(self):
        vid_path = self.data_path / 'Scenes_Dialogues'
        vids = sorted([p.name for p in list(vid_path.glob('Pororo*'))])
        get_starter = Starters('pororoqa')
        prompts = {}

        return vid_path, vids, get_starter, prompts

    def run_movieqa(self):
        vid_path = self.data_path / 'subtt'
        splits = json.load(open(self.data_path / 'MovieQA_benchmark/data/splits.json'))
        vids = splits[self.split]
        get_starter = Starters('movieqa')
        prompts = {}
        return vid_path, vids, get_starter, prompts

    def run_dramaqa(self):
        vid_path = self.data_path
        vids = get_keys_dramaqa(self.data_path, self.split)
        get_starter = Starters('dramaqa')
        prompts = {
            'base_plot': (
                "I am a highly intelligent storytelling bot. "
                "If you give me a description of a scene from a tv series, "
                "I will give you the short synopsis in a couple of sentences.\n"
            ),
        }
        '''
        prompts = {
            'base_plot': (
                "I am a highly intelligent storytelling bot. "
                "If you give me a description of a scene from a tv series, "
                "I will give you the short synopsis in a couple of sentences. \n"
            ),
            'char_desc': (
                "I am a highly intelligent storytelling bot. "
                "If you give me a description of a scene from a tv series, "
                "I will give you the updated character descriptions.\n"
            ),
            'char_plot': (
                "I am a highly intelligent storytelling bot. "
                "If you give me the character description and a description of a scene from a tv series, "
                "I will give you the short synopsis in a couple of sentences. \n"
            ),
        }
        '''
        return vid_path, vids, get_starter, prompts


def load_vids_tvqa(data_path, vid_path):
    vids = list(vid_path.glob('*.srt'))
    vids = ['_'.join(p.stem.split('_')[:-2]) for p in vids]

    # get only vids in val/test set
    qas = load_qa_sets_tvqa(data_path)
    keys = ['_'.join(v['vid_name'].split('_')[:-2]) for v in qas]
    keys = set(keys)
    vids = list(set(vids) & set(keys))

    return vids


def load_qa_sets_tvqa(data_path):
    path = data_path / 'tvqa_qa_release'
    sets = []
    for p in path.glob('*.jsonl'):
        if 'train' in p.stem:
            continue
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                sets.append(row)
    return sets


class LoadSubttBase(DataSpecificsBase):
    def __call__(self, path, key):
        keys, chunks, context = self.run(path, key)
        if len(chunks) != 0:
            if not isinstance(chunks[0], dict):
                chunks = [{'Subtitle': v} for v in chunks]
        return keys, chunks, context


class LoadSubtt(LoadSubttBase):
    def run_tvqa(self, vid_path, key):
        paths = list(vid_path.glob(f'{key}_*'))
        paths = {int(p.stem.split('_')[-1]): p for p in paths}
        keys = sorted(list(paths.keys()))
        subtts = []
        for k in keys:
            v = paths[k]
            subt = load_srt(v)
            subt = [v.content for v in subt if len(v.content) > 0]
            subtts.append(subt)
        return keys, subtts, None

    def run_pororoqa(self, vid_path, key):
        path = vid_path / key
        paths = list(path.glob('*ep*'))
        paths = sorted(paths, key=lambda p: int(p.name.split('_')[-1][2:]))
        keys = []
        subtts = []
        for path in paths:
            with open(path / 'subtitles.txt') as f:
                subtt = f.read().strip()
            subtt = subtt.split('\n')
            subtts.append(subtt)
            key = int(path.name.split('_')[-1][2:])
            keys.append(key)
        return keys, subtts, None

    def init_movieqa(self):
        boundary_path = self.data_path / 'MovieQA_benchmark' / "clip_filenames.txt"
        self.matidx_path = self.data_path / 'matidx'
        self.boundaries = load_scene_boundaries(boundary_path)

    def run_movieqa(self, vid_path, key):
        subt_path = vid_path / f"{key}.srt"
        chunks = load_subtt_movieqa(
            self.args, self.boundaries,
            key, subt_path, self.matidx_path
        )
        return list(range(len(chunks))), chunks, None

    def init_dramaqa(self):
        self.subtitles, self.visuals = get_inits_dramaqa(self.data_path)

    def run_dramaqa(self, vid_path, scene):
        raw_subtt = {}
        if scene in self.subtitles:
            raw_subtt = self.subtitles[scene]

        raw_vis = {}
        if scene in self.visuals:
            raw_vis = self.visuals[scene]
        all_keys = set(raw_subtt.keys()) | set(raw_vis.keys())
        all_keys = sorted(list(all_keys))

        chunks = []
        for key in all_keys:
            row = {}
            if key in raw_subtt:
                st = [v.strip() for v in raw_subtt[key]]
                st = [v for v in st if len(st) > 0]
                if len(st) > 0:
                    row['Subtitle'] = st
            if key in raw_vis:
                vis = [v.strip() for v in raw_vis[key]]
                vis = [v for v in vis if len(vis) > 0]
                if len(vis) > 0:
                    row['Visual'] = vis
            chunks.append(row)
        _keys, _chunks = all_keys, chunks
        keys, chunks = [], []

        def is_different(a, b):
            if 'Subtitle' in a:
                if 'Subtitle' in b:
                    va, vb = a['Subtitle'], b['Subtitle']
                    if len(va) != len(vb):
                        return True
                    else:
                        for vva, vvb in zip(va, vb):
                            if vva[:15] != vvb[:15]:
                                return True
                else:
                    return True
            else:
                if 'Subtitle' in b:
                    return True
            if 'Visual' in a:
                if 'Visual' in b:
                    va, vb = a['Visual'], b['Visual']
                    if len(va) != len(vb):
                        return True
                    else:
                        for vva, vvb in zip(va, vb):
                            if vva[:15] != vvb[:15]:
                                return True
                else:
                    return True
            else:
                if 'Visual' in b:
                    return True
            return False

        last_chunk = None
        for key, chunk in zip(_keys, _chunks):
            if last_chunk is None:
                keys.append(key)
                chunks.append(chunk)
            else:
                if is_different(last_chunk, chunk) and len(chunk) > 0:
                    keys.append(key)
                    chunks.append(chunk)
            last_chunk = chunk

        return keys, chunks, None
