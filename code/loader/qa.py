import random
import re
import csv
import json
from pathlib import Path
from collections import defaultdict

from .base import DataSpecificsBase


def format_gt_index(subtt, gt_index):
    gt_index = int(gt_index)
    if subtt is None:
        return True
    keys, chunks = subtt
    gt_index = [i for i, v in enumerate(keys) if v == gt_index][0]
    return gt_index


class DataSpecificsQA(DataSpecificsBase):
    def __call__(self):
        vid_path, qa, gt_plots, get_vid_key, get_gt_index, unload_data = \
            self.run()
        # improve estimatability of accuracy of the main set given a temporal subset
        random.shuffle(qa)
        return vid_path, qa, gt_plots, get_vid_key, get_gt_index, unload_data

    def run_tvqa(self):
        return Loader_tvqa.run(self.split, self.data_path)

    def run_pororoqa(self):
        return Loader_pororoqa.run(self.split, self.data_path)

    def run_movieqa(self):
        return Loader_movieqa.run(self.split, self.data_path)

    def run_dramaqa(self):
        return Loader_dramaqa.run(self.split, self.data_path)


class Loader_tvqa:
    def __init__(self, split, data_path):
        name = split
        if name == 'test':
            name = 'test_public'
        self.qa_path = data_path / 'tvqa_qa_release' / f'tvqa_{name}.jsonl'
        assert self.qa_path.is_file()
        self.vid_path = data_path / 'tvqa_subtitles'

        with open(self.qa_path) as f:
            qa = []
            for line in f:
                qa.append(json.loads(line))
        self.qa = qa
        self.gt_plots = None

    @classmethod
    def run(cls, split, data_path):
        obj = cls(split, data_path)
        return obj.vid_path, obj.qa, obj.gt_plots, obj.get_vid_key, obj.get_gt_index, obj.unload_data

    @staticmethod
    def get_vid_key(q_):
        key = q_['vid_name']
        key = '_'.join(key.split('_')[:-2])
        return key

    @staticmethod
    def get_gt_index(q_, subtt=None):
        gt_index = q_['vid_name'].split('_')[-1]
        return format_gt_index(subtt, gt_index)

    def unload_data(self, q_):
        question = q_['q']
        answers = [q_[f'a{i}'] for i in range(5)]
        qid = q_['qid']
        key = self.get_vid_key(q_)

        answer_key = 'answer_idx'
        answer = None
        if answer_key in q_:
            answer = q_[answer_key]
        return qid, question, answers, answer, key


class Loader_pororoqa(Loader_tvqa):
    def __init__(self, split, data_path):
        self.vid_path = data_path / 'Scenes_Dialogues'
        self.qa_path = data_path / 'qa.json'
        self.gt_plot_path = data_path / 'descriptions.csv'

        with open(self.qa_path) as f:
            qa = json.load(f)['PororoQA']

        num = len(qa) // 10
        valqa = qa[num * 6: num * 8]
        testqa = qa[num * 8:]

        qa = testqa if split == 'test' else valqa

        with open(self.gt_plot_path) as f:
            gt_plots = defaultdict(lambda: {})
            reader = csv.reader(f)
            for line in reader:
                key, num, desc = line
                desc = desc.strip().replace('\n', ' ')
                gt_plots[key][num] = desc
        temp = defaultdict(lambda: {})
        for k, v in gt_plots.items():
            v = {int(k2): v2 for k2, v2 in v.items()}
            nums = sorted(list(v.keys()))
            descs = [v[n] for n in nums]
            descs = ' '.join(descs)
            descs = '.'.join(descs.split('.')[:5])
            k = k.split('_')
            vid = '_'.join(k[:-1])
            epi = int(k[-1][2:])

            temp[vid][epi] = descs
        gt_plots = dict(temp)

        self.qa = qa
        self.gt_plots = gt_plots

    @staticmethod
    def get_vid_key(q_):
        key = q_['video_name']
        key = '_'.join(key.split('_')[:-1])
        return key

    @staticmethod
    def get_gt_index(q_, subtt=None):
        gt_index = q_['video_name'].split('_')[-1][2:]
        gt_index = int(re.findall(r'\d+', gt_index)[0])
        return format_gt_index(subtt, gt_index)

    def unload_data(self, q_):
        question = q_['question']
        answers = [q_[f'answer{i}'] for i in range(5)]
        qid = q_['qid']
        key = self.get_vid_key(q_)

        answer_key = 'correct_idx'
        answer = None
        if answer_key in q_:
            answer = q_[answer_key]
        return qid, question, answers, answer, key


class Loader_movieqa(Loader_tvqa):
    def __init__(self, split, data_path):
        self.vid_path = data_path / 'subtt'
        qa = json.load(open(data_path / 'MovieQA_benchmark/data/qa.json','r'))
        valqa = qa[-5096:-3138]
        testqa = qa[-3138:]

        qa = testqa if split == 'test' else valqa

        '''
        movies = json.load(open(self.data_path / 'MovieQA_benchmark/data/movies.json'))
        splits = json.load(open(self.data_path / 'MovieQA_benchmark/data/splits.json'))
        '''

        gt_plots = None  # TODO

        self.qa = qa
        self.gt_plots = gt_plots

    @staticmethod
    def get_vid_key(q_):
        key = q_['imdb_key']
        return key

    @staticmethod
    def get_gt_index(q_, subtt=None):
        res = q_['video_clips']
        if len(res) == 0:
            res = None
        return res

    def unload_data(self, q_):
        question = q_['question']
        answers = q_['answers']
        qid = q_['qid']
        key = self.get_vid_key(q_)

        answer_key = 'correct_index'
        answer = None
        if answer_key in q_:
            answer = q_[answer_key]
        return qid, question, answers, answer, key


class Loader_dramaqa(Loader_tvqa):
    def __init__(self, split, data_path):
        name = split
        self.vid_path = data_path
        qa = json.load(open(data_path / f'DramaQA/past/AnotherMissOhQA_{name}_set.json','r'))

        gt_plots = None

        self.qa = qa
        self.gt_plots = gt_plots

    @staticmethod
    def get_vid_key(q_):
        scene = q_['vid'].split('_')[:2]
        scene = '_'.join(scene)
        return scene

    @staticmethod
    def get_gt_index(q_, subtt=None):
        shots = q_['shot_contained']
        if subtt is None:
            # return if gt index is available when given subtt
            return True if len(shots) > 0 else None
        keys, chunks = subtt

        shots = [int(v) for v in shots]
        keys = [int(v) for v in keys]
        if len(keys) == 1:
            res = keys
        else:
            if len(shots) == 1:
                res = shots
            elif len(shots) == 2:
                st, et = shots
                res = [v for v in keys if v >= st and v <= et]
        res = set(res)
        res = [i for i, k in enumerate(keys) if k in res]

        return res

    def unload_data(self, q_):
        question = q_['que']
        answers = q_['answers']
        qid = q_['qid']
        key = self.get_vid_key(q_)

        answer_key = 'correct_idx'
        answer = None
        if answer_key in q_:
            answer = q_[answer_key]
        return qid, question, answers, answer, key
