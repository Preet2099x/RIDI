"""Loads the pretrained RIDI SVR cascade (OpenCV YAML) and predicts without OpenCV.

The YAML files are ~2 GB in total, so they are parsed once and cached as a
compact .npz next to the model folder.
"""
import os
import re
import numpy as np

MODEL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ridi_imu_models', 'svr_cascade0308'))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')


class SVM:
    def __init__(self, svm_type, gamma, sv, dfs, n_class):
        self.svm_type = svm_type          # 'C_SVC' or 'EPS_SVR'
        self.gamma = gamma
        self.sv = sv                      # (n_sv, 1200) float32
        self.sv_sq = (sv.astype(np.float64) ** 2).sum(1)
        self.dfs = dfs                    # list of (rho, alpha, index)
        self.n_class = n_class

    def _kernel(self, x):
        dots = (self.sv @ x.astype(np.float32)).astype(np.float64)
        d2 = self.sv_sq + float(x @ x) - 2.0 * dots
        return np.exp(-self.gamma * d2)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        k = self._kernel(x)
        if self.svm_type == 'EPS_SVR':
            rho, alpha, idx = self.dfs[0]
            return float(alpha @ k[idx] - rho)
        votes = np.zeros(self.n_class, dtype=int)
        n = 0
        for i in range(self.n_class):
            for j in range(i + 1, self.n_class):
                rho, alpha, idx = self.dfs[n]
                n += 1
                s = float(alpha @ k[idx] - rho)
                votes[i if s > 0 else j] += 1
        return int(np.argmax(votes))


def _floats(text):
    text = text.replace('[', ' ').replace(']', ' ').replace(',', ' ')
    return np.fromstring(text, sep=' ')


def _parse_yaml(path):
    with open(path, 'r') as f:
        text = f.read()
    svm_type = re.search(r'svmType:\s*(\w+)', text).group(1)
    gamma = float(re.search(r'gamma:\s*([-+0-9.eE]+)', text).group(1))
    m = re.search(r'class_count:\s*(\d+)', text)
    n_class = int(m.group(1)) if m else 1
    var_count = int(re.search(r'var_count:\s*(\d+)', text).group(1))
    a = text.index('support_vectors:') + len('support_vectors:')
    b = text.index('decision_functions:')
    sv = _floats(text[a:b].replace('- [', ' ')).astype(np.float32).reshape(-1, var_count)
    tail = text[b:]
    dfs = []
    for blk in tail.split('sv_count:')[1:]:
        rho = float(re.search(r'rho:\s*([-+0-9.eE]+)', blk).group(1))
        if 'index:' in blk:
            alpha = _floats(re.search(r'alpha:(.*?)index:', blk, re.S).group(1))
            idx_txt = blk.split('index:')[1]
            idx = _floats(idx_txt[:idx_txt.rindex(']') + 1]).astype(np.int64)
        else:
            alpha_txt = blk.split('alpha:')[1]
            alpha = _floats(alpha_txt[:alpha_txt.rindex(']') + 1])
            idx = np.arange(len(alpha))
        dfs.append((rho, alpha, idx))
    return svm_type, gamma, sv, dfs, n_class


def _load_one(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, name + '.npz')
    if not os.path.exists(cache):
        print('First run: converting %s (one time, about a minute)...' % name, flush=True)
        svm_type, gamma, sv, dfs, n_class = _parse_yaml(os.path.join(MODEL_DIR, name + '.yaml'))
        arrs = {'sv': sv, 'meta': np.array([gamma, n_class]), 'type': np.array(svm_type),
                'rho': np.array([d[0] for d in dfs])}
        for i, d in enumerate(dfs):
            arrs['alpha%d' % i] = d[1]
            arrs['idx%d' % i] = d[2]
        np.savez(cache, **arrs)
    z = np.load(cache)
    n = len(z['rho'])
    dfs = [(float(z['rho'][i]), z['alpha%d' % i], z['idx%d' % i]) for i in range(n)]
    return SVM(str(z['type']), float(z['meta'][0]), z['sv'], dfs, int(z['meta'][1]))


class SVRCascade:
    """Classifier (placement) + 2 regressors (local speed x, z) per placement."""

    def __init__(self, path=MODEL_DIR):
        with open(os.path.join(path, 'class_map.txt')) as f:
            tok = f.read().split()
        n = int(tok[0])
        self.class_names = [''] * n
        for i in range(n):
            self.class_names[int(tok[2 + 2 * i])] = tok[1 + 2 * i]
        self.n_class = n
        self.classifier = _load_one('classifier')
        self.regressors = [[_load_one('regressor_%d_%d' % (c, k)) for k in range(2)] for c in range(n)]

    def class_id(self, name):
        return self.class_names.index(name)

    def predict(self, feature, force_label=None):
        label = self.classifier.predict(feature) if force_label is None else force_label
        r = self.regressors[label]
        return label, np.array([r[0].predict(feature), r[1].predict(feature)])
