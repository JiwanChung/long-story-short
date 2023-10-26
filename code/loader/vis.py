import json
import pickle
from collections import defaultdict

import numpy as np

from .qa import DataSpecificsQA
from .base import LoadSubtt


class DataSpecificsVis(DataSpecificsQA):
    def __init__(self, args, data_path, dataset: str = 'tvqa'):
        super().__init__(args, data_path, dataset)

        self.run_vis = getattr(self, f'run_vis_{self.dataset}')
        if hasattr(self, f'init_vis_{self.dataset}'):
            getattr(self, f"init_vis_{self.dataset}")()

    def __call__(self):
        self.load_subtt = LoadSubtt(self.args, self.data_path, self.dataset)
        res = self.run()
        self.vid_path = res[0]
        return [*res, self.run_vis()]

    def init_vis_movieqa(self):
        matidx_path = self.data_path / 'matidx'
        vis_path = self.data_path / 'frame_clip'
        vis_keys = defaultdict(lambda: [])
        for path in vis_path.glob('*.pkl'):
            name = path.name.split('.')
            key = name[0]
            vis_keys[key].append(path)
        self.vis_keys = dict(vis_keys)

    def run_vis_movieqa(self):

        def load_vis(key):
            if key not in self.vis_keys:
                return None
            boundary = self.load_subtt.boundaries[key]
            ets = sorted([v[1] for v in boundary])
            vis = []
            vis_keys = []
            for path in self.vis_keys[key]:
                st, et = path.stem.split('.')[1:3]
                st, et = int(st[3:]), int(et[3:])
                with open(path, 'rb') as f:
                    x = pickle.load(f)
                    x = x['features']

                res = None
                for i, bet in enumerate(ets):
                    if et <= bet:
                        res = i
                        break

                if res is None:
                    res = len(ets) - 1

                if len(x) > 0:
                    vis.append(x)
                    vis_keys.append(i)

            return vis, vis_keys

        return load_vis

    def init_vis_dramaqa(self):
        self.vis_path = self.data_path / 'DramaQA/frame_clip'
        vis_keys = defaultdict(lambda: [])
        for path in self.vis_path.glob('*.pkl'):
            name = path.stem
            ep, scene, shot = name.split('_')
            key = '_'.join([ep, scene])
            vis_keys[key].append(shot)
        self.vis_keys = {k: sorted(v, key=lambda x: int(x)) for k, v in vis_keys.items()}

    def run_vis_dramaqa(self):

        def load_vis(key):
            txt_keys, _, _ = self.load_subtt(self.vid_path, key)
            txt_keys = [int(v) for v in txt_keys]
            shots = set([int(v) for v in self.vis_keys[key]])

            vis = []
            vis_keys = []
            for i, shot in enumerate(txt_keys):
                if shot in shots:
                    fname = f'{key}_{shot:04d}'
                    path = self.vis_path / f'{fname}.pkl'
                    with open(path, 'rb') as f:
                        x = pickle.load(f)
                    img_keys = sorted(list(x.keys()), key=lambda v: int(v.split('_')[1]))
                    imgs = [x[k] for k in img_keys]
                    imgs = np.stack(imgs, axis=0)
                    if len(imgs) > 0:
                        vis.append(imgs)
                        vis_keys.append(i)
                else:
                    import ipdb; ipdb.set_trace()  # XXX DEBUG
            return vis, vis_keys

        return load_vis

    def init_vis_pororoqa(self):
        self.vis_path = self.data_path / 'frame_clip'
        vis_keys = defaultdict(lambda: [])

        for path in self.vis_path.glob('*.pkl'):
            name = path.stem
            name = name.split('_')
            season = name[:-1]

            # removing duplicates
            temp = []
            for word in season:
                if word not in temp:
                    temp.append(word)
            season = '_'.join(temp)

            epi = int(name[-1][2:])
            vis_keys[season].append(epi)

        self.vis_keys = {k: sorted(v, key=lambda x: int(x)) for k, v in vis_keys.items()}

    def run_vis_pororoqa(self):

        def load_vis(key):

            vis_keys = self.vis_keys[key]

            vis = []
            for k in vis_keys:
                fname = f'{key}_{key}_ep{shot}'
                path = self.vis_path / f'{fname}.pkl'
                with open(path, 'rb') as f:
                    x = pickle.load(f)
                img_keys = sorted(list(x.keys()), key=lambda v: int(v))
                imgs = [x[k] for k in img_keys]
                imgs = np.stack(imgs, axis=0)
                vis.append(imgs)
            return vis, vis_keys

        return load_vis
