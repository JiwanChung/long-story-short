import time
from collections import defaultdict

import openai
from rouge import Rouge
import numpy as np


def run_with_tries(args, prompt, max_tries: int = 10, **kwargs):
    response = None
    e_ = None
    for i in range(max_tries):
        try:
            response = openai.Completion.create(
                engine=args.engine,
                prompt=prompt,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                **kwargs,
            )
            return response
        except Exception as e:
            e_ = e
            time.sleep(5)
            continue
    if response is None:
        print(e_)
        raise Exception('OpenAI API Error')


def get_chunks(li, size):
    return [li[i:i + size] for i in range(0, len(li), size)]


def get_seq_unique(txt):
    if isinstance(txt, str):
        txt = txt.split('\n')
    if len(txt) == 0:
        return ''
    res = [txt[0]]
    if len(txt) > 1:
        for row in txt[1:]:
            if row != res[-1]:
                res.append(row.strip())
    return res


class IndexFinder:
    def __init__(self):
        self.rouge = Rouge()

    def __call__(self, story, hypo):
        plots = story[story.find('Plot:'):]
        plots = [' '.join(v.split()[1:]).strip() for v in plots.strip().split('\n')[1:]]
        try:
            scores = [self.rouge.get_scores(hypo, v)[0]["rouge-l"]['r'] for v in plots]
            index = np.argmax(scores)
            return int(index)
        except Exception as e:
            print(f'Error in metric-based index finding: {e}')
            return None


def get_unique_ordered(li):
    res = []
    for row in li:
        if row not in res:
            res.append(row)
    return res


def format_context(c_, gap='\n\n'):
    subtt = ''
    vkeys = {'Visual': 'Scene', 'Subtitle': 'Subtitle'}
    for cname, val in vkeys.items():
        if cname in c_:
            cs = c_[cname]
            cs = [v.strip() for v in cs]
            cs = [v[1:] if v.startswith('-') else v for v in cs]
            cs = [f'{v.strip()} ' for v in cs]
            # cs = [f'- {v}' if not v.startswith('- ') else v for v in cs]
            subtt += f'{val}:\n' + '\n'.join(cs) + gap
    subtt = subtt.strip()
    return subtt


def format_context_li(chunks, gap='\n\n'):
    subtt = ''
    res = defaultdict(list)
    vkeys = {'Visual': 'Scene Description', 'Subtitle': 'Subtitle'}
    for cname, val in vkeys.items():
        for c_ in chunks:
            if cname in c_:
                cs = c_[cname]
                cs = [v.strip() for v in cs]
                '''
                cs = [v[1:] if v.startswith('-') else v for v in cs]
                cs = [f'{v.strip()} ' for v in cs]
                '''
                cs = [f'- {v}' if not v.startswith('- ') else v for v in cs]
                res[cname].extend(cs)

    subtt = ''
    for cname, val in vkeys.items():
        cs = res[cname]
        cs = [v.strip() for v in cs]
        cs = [f'- {v}' if not v.startswith('- ') else v for v in cs]
        if cname == 'Visual':
            cs = get_unique_ordered(cs)
            # cs = [f'{v.strip()} ' for v in cs]
        else:
            cs = get_seq_unique(cs)
        cs = [v.strip() for v in cs]
        cs = '\n'.join(cs)
        cs = cs.strip()
        if len(cs) > 0:
            subtt += f'{val}:\n' + cs + gap
    subtt = subtt.strip()
    return subtt


def softmax(v, temp=1, eps=1e-10):
    v = v / temp
    ve = np.exp(v)
    return ve / (ve.sum(axis=-1, keepdims=True) + eps)


def get_ent(x):
    x = np.array(x)
    return sum(-x * np.log(x))


def get_bent(x):
    x = np.array(x)
    x = np.sort(x)[-2:]
    x = x / x.sum()
    return sum(-x * np.log(x))


def get_bmask(x):
    v = np.sort(x)
    return x >= v[-2]
