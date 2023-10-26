class Starters:
    def __init__(self, name):
        self.get_starter = getattr(self, f'get_starter_{name}', self.get_starter_base)

    def __call__(self, key):
        return self.get_starter(key)

    @staticmethod
    def get_starter_base(key):
        starter = '1.'
        return starter

    @staticmethod
    def get_starter_pororoqa(key):
        starter = '1. Pororo:'
        return starter

    '''
    @staticmethod
    def get_starter_dramaqa(key):
        starter = '1. Haeyoung1:'
        return starter
    '''
