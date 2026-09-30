"""Python port of the RIDI C++ localization (imu_localization.cc + imu_optimization.h).

Ceres is not needed: the bias-estimation problem is linear in the biases, so the
same minimum is found with one linear least-squares solve.

Conventions (same as the RIDI dataset / Android):
  time in seconds, gyro rad/s, linacce m/s^2, gravity vector points UP in the
  sensor frame (about +9.8 on the axis pointing at the sky), orientation is a
  (w, x, y, z) quaternion that rotates sensor vectors into a z-up world frame.
"""
import numpy as np

WINDOW = 200            # frames per regression window
REG_INTERVAL = 30       # frames between regressions
SIGMA = 2.0             # feature smoothing
LOCAL_OPT_INTERVAL = 200
LOCAL_OPT_WINDOW = 1000
MAX_TRANSITION = 2
LABEL_RADIUS = 7
LOCAL_GRAVITY = np.array([0.0, 1.0, 0.0])


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_between(a, b):
    """Shortest-arc rotation matrix taking direction a onto direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(a @ b)
    if c < -1 + 1e-9:
        # 180 degrees: rotate about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.cross(a, b)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _gauss_kernel(sigma, n=17):
    x = np.arange(n) - (n - 1) / 2.0
    k = np.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


_KERNEL = _gauss_kernel(SIGMA)


def make_feature(gyro, linacce, gravity):
    """(200,3) x3 -> 1200-vector, same as ComputeDirectFeatureGravity."""
    n = len(gyro)
    ag = gyro.copy()
    al = linacce.copy()
    gn = np.linalg.norm(gravity, axis=1)
    dot = gravity[:, 1] / np.maximum(gn, 1e-12)
    for i in range(n):
        if dot[i] < -1 + 1e-3:
            ag[i, 1] *= -1
            al[i, 1] *= -1
        elif abs(dot[i]) < 1 - 1e-3:
            R = rot_between(gravity[i], LOCAL_GRAVITY)
            ag[i] = R @ gyro[i]
            al[i] = R @ linacce[i]
    f = np.concatenate([ag, al], axis=1)                    # (n, 6)
    pad = len(_KERNEL) // 2
    fp = np.pad(f, ((pad, pad), (0, 0)), mode='symmetric')
    out = np.zeros_like(f)
    for j, w in enumerate(_KERNEL):
        out += w * fp[j:j + n]
    return out.reshape(-1).astype(np.float32)


def _sizes(n):
    if 600 <= n < 800:
        return 12, 8
    if 800 <= n < 1000:
        return 16, 12
    if 1000 <= n < 5000:
        return 20, 16
    if 5000 <= n < 10100:
        return 100, 96
    return 100, 200


class Trajectory:
    """Online RIDI. Feed frames with add_record(); call finish() at the end."""

    def __init__(self, model, force_label=None, weight_ls=10.0, weight_vs=1.0,
                 init_pos=(0.0, 0.0, 0.0), verbose=False):
        self.model = model
        self.force_label = force_label
        self.wls = np.sqrt(weight_ls)
        self.wvs = np.sqrt(weight_vs)
        self.verbose = verbose
        self.t, self.gyro, self.lin, self.grav = [], [], [], []
        self.R, self.R_GW = [], []
        self.speed = []
        self.pos = []
        self.init_pos = np.array(init_pos, float)
        self.cons_ind, self.cons_ls, self.labels = [], [], []
        self.last_cons = WINDOW - REG_INTERVAL

    @property
    def n(self):
        return len(self.t)

    # ---------------------------------------------------------------- input
    def add_record(self, t, gyro, linacce, gravity, quat):
        quat = np.asarray(quat, float)
        R = quat if quat.shape == (3, 3) else quat_to_mat(quat)
        self.t.append(float(t))
        self.gyro.append(np.asarray(gyro, float))
        self.lin.append(np.asarray(linacce, float))
        self.grav.append(np.asarray(gravity, float))
        self.R.append(R)
        self.R_GW.append(rot_between(np.asarray(gravity, float), LOCAL_GRAVITY) @ R.T)
        if self.n > 1:
            dt = self.t[-1] - self.t[-2]
            s = R @ self.lin[-1] * dt
            self.speed.append(s)
            self.pos.append(s * dt)
        else:
            self.speed.append(np.zeros(3))
            self.pos.append(self.init_pos.copy())
        i = self.n - 1
        if i > LOCAL_OPT_WINDOW and i % LOCAL_OPT_INTERVAL == 0:
            self.run_optimization(i - LOCAL_OPT_WINDOW, LOCAL_OPT_WINDOW)

    def finish(self):
        """Global optimization over the whole recording."""
        if self.n >= 600:
            self.run_optimization(0, self.n)
        return self.positions()

    def positions(self):
        return np.array(self.pos)

    # ----------------------------------------------------------- regression
    def regress_speed(self, end):
        old = len(self.labels)
        for i in range(self.last_cons + REG_INTERVAL, end, REG_INTERVAL):
            sl = slice(i - WINDOW, i)
            feat = make_feature(np.array(self.gyro[sl]), np.array(self.lin[sl]), np.array(self.grav[sl]))
            label, reg = self.model.predict(feat, self.force_label)
            self.labels.append(label)
            self.cons_ind.append(i)
            self.cons_ls.append(np.array([reg[0], 0.0, reg[1]]))
        for i in range(max(old - LABEL_RADIUS, 0), len(self.labels)):
            a = max(i - LABEL_RADIUS, 0)
            b = min(i + LABEL_RADIUS, len(self.labels))
            trans = sum(1 for j in range(a + 1, b) if self.labels[j] != self.labels[j - 1])
            if trans > MAX_TRANSITION:
                self.cons_ls[i] = np.zeros(3)
        if self.cons_ind:
            self.last_cons = self.cons_ind[-1]

    # --------------------------------------------------------- optimization
    def run_optimization(self, start, N):
        self.regress_speed(start + N)
        K, C = _sizes(N)
        C = min(C, len(self.cons_ind))
        if C < 2:
            return
        if start > 0:
            sel = list(range(len(self.cons_ind) - C, len(self.cons_ind)))
            init_speed = self.speed[start].copy()
        else:
            inc = len(self.cons_ind) / (C - 1.0)
            sel = [int(i * inc) for i in range(C - 1)] + [len(self.cons_ind) - 1]
            init_speed = np.zeros(3)
        cind = np.array([self.cons_ind[i] - start for i in sel])
        ls_t = np.array([self.cons_ls[i] for i in sel])
        keep = (cind >= 1) & (cind < N)
        cind, ls_t = cind[keep], ls_t[keep]

        sl = slice(start, start + N)
        t = np.array(self.t[sl])
        lin = np.array(self.lin[sl])
        R = np.array(self.R[sl])
        RGW = np.array(self.R_GW[sl])

        # sparse grid
        interval = N // K
        vi = np.array([(i + 1) * interval - 1 for i in range(K)])
        alpha = np.zeros(N)
        inv = np.zeros(N, dtype=int)
        alpha[:vi[0] + 1] = (t[:vi[0] + 1] - t[0]) / (t[vi[0]] - t[0])
        for i in range(1, K):
            j = np.arange(vi[i - 1] + 1, vi[i] + 1)
            inv[j] = i
            alpha[j] = (t[j] - t[vi[i - 1]]) / (t[vi[i]] - t[vi[i - 1]])
        W = np.zeros((N, K))
        rows = np.arange(N)
        W[rows, inv] += alpha
        m = inv > 0
        W[rows[m], inv[m] - 1] += 1 - alpha[m]

        dt = np.zeros(N)
        dt[1:] = t[1:] - t[:-1]
        # speed[i] = s0 + sum_{j<i} dt[j+1] * R[j] (lin[j] + W[j] b)   (dt of step j -> j+1)
        step = np.zeros(N)
        step[:-1] = dt[1:]
        base_inc = np.einsum('nij,nj->ni', R, lin) * step[:, None]
        base = np.vstack([np.zeros(3), np.cumsum(base_inc, axis=0)[:-1]]) + init_speed

        nc = len(cind)
        J = np.zeros((3 * nc, 3 * K))          # rows: [ls_x | vs | ls_z] blocks
        # d speed[ind, d] / d b[c, k]
        Jsp = np.zeros((3, 3, nc, K))          # [d][c][constraint][k]
        for d in range(3):
            for c in range(3):
                term = (step * R[:, d, c])[:, None] * W
                cs = np.vstack([np.zeros((1, K)), np.cumsum(term, axis=0)[:-1]])
                Jsp[d, c] = cs[cind]
        sp0 = base[cind]
        Rc = RGW[cind]
        r0 = np.zeros(3 * nc)
        for c in range(3):
            sl_c = slice(c * K, (c + 1) * K)
            J[0:nc, sl_c] = self.wls * np.einsum('nd,dnk->nk', Rc[:, 0, :], Jsp[:, c])
            J[nc:2 * nc, sl_c] = self.wvs * Jsp[2, c]
            J[2 * nc:3 * nc, sl_c] = self.wls * np.einsum('nd,dnk->nk', Rc[:, 2, :], Jsp[:, c])
        ls0 = np.einsum('nij,nj->ni', Rc, sp0)
        r0[0:nc] = self.wls * (ls0[:, 0] - ls_t[:, 0])
        r0[nc:2 * nc] = self.wvs * sp0[:, 2]
        r0[2 * nc:] = self.wls * (ls0[:, 2] - ls_t[:, 2])
        b = np.linalg.solve(J.T @ J + np.eye(3 * K), -J.T @ r0)
        if self.verbose:
            print('opt start=%d N=%d K=%d C=%d cost %.3f -> %.3f' % (
                start, N, K, nc, r0 @ r0 + 0, np.sum((J @ b + r0) ** 2) + b @ b))

        # commit: correct acceleration, redo integration
        corr = W @ b.reshape(3, K).T           # (N, 3)
        for j in range(N):
            self.lin[start + j] = self.lin[start + j] + corr[j]
        self._reintegrate(start)

    def _reintegrate(self, start):
        n = self.n
        if start + 1 >= n:
            return
        t = np.array(self.t[start:])
        lin = np.array(self.lin[start:])
        R = np.array(self.R[start:])
        dt = t[1:] - t[:-1]
        inc = np.einsum('nij,nj->ni', R[:-1], lin[:-1]) * dt[:, None]
        sp = self.speed[start] + np.cumsum(inc, axis=0)
        pinc = np.vstack([self.speed[start][None, :], sp[:-1]]) * dt[:, None]
        ps = self.pos[start] + np.cumsum(pinc, axis=0)
        for k in range(n - start - 1):
            self.speed[start + 1 + k] = sp[k]
            self.pos[start + 1 + k] = ps[k]
