import torch
from torch.utils.data import Sampler


class StatefulShuffleSampler(Sampler):
    """Deterministic shuffle sampler with resumable permutation."""

    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = int(seed) % (2 ** 32)
        self.epoch = 0
        self._perm = None
        self._perm_epoch = None

    def set_epoch(self, epoch):
        epoch = int(epoch)
        self.epoch = epoch
        if self._perm_epoch != epoch:
            self._perm = None
            self._perm_epoch = epoch

    def _generate_perm(self):
        g = torch.Generator()
        g.manual_seed((self.seed + self.epoch) % (2 ** 32))
        self._perm = torch.randperm(len(self.data_source), generator=g).tolist()
        self._perm_epoch = self.epoch

    def __iter__(self):
        if self._perm is None or self._perm_epoch != self.epoch:
            self._generate_perm()
        return iter(self._perm)

    def __len__(self):
        return len(self.data_source)

    def state_dict(self):
        if self._perm is None or self._perm_epoch != self.epoch:
            self._generate_perm()
        return {
            'seed': self.seed,
            'epoch': self.epoch,
            'perm': torch.tensor(self._perm, dtype=torch.int64),
        }

    def load_state_dict(self, state):
        if not state:
            return
        if state.get('seed') is not None:
            self.seed = int(state['seed']) % (2 ** 32)
        if state.get('epoch') is not None:
            self.epoch = int(state['epoch'])
        perm = state.get('perm')
        if perm is None:
            self._perm = None
            self._perm_epoch = None
            return
        if torch.is_tensor(perm):
            perm = perm.cpu().tolist()
        if len(perm) != len(self.data_source):
            self._perm = None
            self._perm_epoch = None
            return
        self._perm = perm
        self._perm_epoch = self.epoch
