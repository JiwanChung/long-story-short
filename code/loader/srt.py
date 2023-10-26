import srt


def load_srt(p):
    err = None
    for encoding in ['iso-8859-1', 'UTF8', 'iso-8859-2']:
        try:
            res = load_srt_encoding(p, encoding)
            return res
        except Exception as e:
            err = e
    raise Exception(err)


def load_srt_encoding(p, encoding='iso-8859-1'):
    with open(p, encoding=encoding) as f:
        subt = f.readlines()
    subt = list(srt.parse(''.join(subt)))
    return subt
