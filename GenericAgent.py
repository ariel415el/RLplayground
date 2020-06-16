
class GenericAgent(object):
    def __init__(self,train=True):
        self.train=train
        self.reporter=None

    def process_new_state(self, state):
        raise NotImplementedError

    def process_output(self, new_state, reward, is_finale_state):
        raise NotImplementedError

    def load_state(self, path):
        raise NotImplementedError

    def save_state(self, path):
        raise NotImplementedError

    def get_stats(self):
        raise NotImplementedError

    def train(self):
        self.train = True

    def eval(self):
        self.train = False

    def set_reporter(self, reporter):
         self.reporter = reporter