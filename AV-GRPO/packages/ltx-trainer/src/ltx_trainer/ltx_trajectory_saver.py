import os
import torch
from collections import defaultdict

class TrajectoryCollector:
    def __init__(self, indices='all', total_steps=0, store_as_compact=True):
        self.indices = indices
        self.total_steps = total_steps
        self.store_as_compact = store_as_compact
        self._target_set = self._normalize_indices()
        self._collected_data = defaultdict(list)
        self._collected_indices = []

    def _normalize_indices(self):
        if self.indices is None:
            return set()
        if self.indices == 'all':
            return None
        total_positions = self.total_steps + 1
        norm = set()
        for idx in self.indices:
            if idx < 0:
                idx = total_positions + idx
            if 0 <= idx < total_positions:
                norm.add(idx)
        return norm

    @property
    def is_disabled(self):
        return self._target_set is not None and len(self._target_set) == 0

    @property
    def collect_all(self):
        return self._target_set is None

    def should_collect(self, step_idx):
        if self.is_disabled:
            return False
        if self.collect_all:
            return True
        return step_idx in self._target_set

    def collect(self, step_idx, **kwargs):
        if not self.should_collect(step_idx):
            return
        self._collected_indices.append(step_idx)
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                self._collected_data[key].append(value)

    def get_result(self):
        return dict(self._collected_data)

    def get_index_map(self):
        if self.is_disabled or not self.store_as_compact:
            return None
        total_positions = self.total_steps + 1
        if self.collect_all:
            return torch.arange(total_positions, dtype=torch.long)
        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        for compact_idx, orig_idx in enumerate(self._collected_indices):
            index_map[orig_idx] = compact_idx
        return index_map

    @property
    def collected_indices(self):
        return self._collected_indices

    def reset(self):
        self._collected_data = defaultdict(list)
        self._collected_indices = []

    def __len__(self):
        return len(self._collected_indices)

    def save(self, save_path, extra_metadata=None):
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        save_dict = {
            'total_steps': self.total_steps,
            'collected_indices': self.collected_indices,
            'index_map': self.get_index_map(),
            'data': self.get_result(),
        }
        if extra_metadata:
            save_dict.update(extra_metadata)
        torch.save(save_dict, save_path)
        print(f"Trajectory saved to {save_path}")