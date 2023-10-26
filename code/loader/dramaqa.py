import json
from collections import defaultdict


def get_keys_dramaqa(data_path, split: str = 'val'):
    # only extract shots necessary for QA {split} set
    qa = json.load(open(data_path / f'DramaQA/AnotherMissOhQA_{split}_set.json','r'))
    qa_keys = set()
    for row in qa:
        key = row['vid']
        ep, scene, shot = key.split('_')
        scene = '_'.join([ep, scene])
        qa_keys.add(scene)
    keys = sorted(list(qa_keys))
    return keys


def load_data_dramaqa(data, func):
    res = defaultdict(dict)
    for name, shot in data.items():
        episode, scene, shot_name = name.split('_')
        subs = func(shot)
        subs = remove_duplicate_ordered(subs)
        res[f'{episode}_{scene}'][int(shot_name)] = subs
    return res


def get_inits_dramaqa(data_path):
    subtitles = json.load(open(data_path / 'DramaQA' / 'AnotherMissOh_script.json'))
    visuals = json.load(open(data_path / 'DramaQA' / 'AnotherMissOh_Visual.json'))

    subtitles = load_data_dramaqa(subtitles, format_script)
    visuals = load_data_dramaqa(visuals, format_visual)
    return subtitles, visuals


def format_script(dt):
    li = dt['contained_subs']
    res = []
    for line in li:
        x = line['speaker']
        y = line['utter']
        txt = f'{x}: {y}'
        res.append(txt)
    return res


def format_visual(li):
    res = []
    for line in li:
        x = line['persons']
        for x2 in x:
            txt = format_visual_(x2['person_id'], x2['person_info']['behavior'], x2['person_info']['emotion'])
            res.append(txt)
    return res


def format_visual_(name, behab, emo):
    return f'{name} {behab}, feeling {emo}.'


def remove_duplicate_ordered(data):
    res = []
    keys = set()
    for line in data:
        if line not in keys:
            res.append(line)
            keys.add(line)
    return res
