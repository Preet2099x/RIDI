"""Runs RIDI on a converted recording and shows the path.

    python run_ridi.py recording_ridi.npz [--placement leg|handheld|bag|body|auto] [--no-show]
"""
import argparse
import os
import time

import numpy as np

import ridi_core as rc
import ridi_model


def run(data, model, placement='leg', verbose=False):
    label = None if placement == 'auto' else model.class_id(placement)
    tr = rc.Trajectory(model, force_label=label, verbose=verbose)
    for i in range(len(data['t'])):
        tr.add_record(data['t'][i], data['gyro'][i], data['lin'][i], data['grav'][i], data['quat'][i])
    pos = tr.finish()
    return pos, tr


def close_loop(pos):
    """For walks that end where they started: spread the end-to-start gap over the path.

    The gap is removed in proportion to the distance walked, so standing still
    doesn't absorb any of the correction. RIDI's own drift is slow and smooth,
    which makes this a good fit for the error.
    """
    step = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    if s[-1] < 1e-6:
        return pos.copy()
    gap = pos[-1] - pos[0]
    return pos - (s / s[-1])[:, None] * gap


def summarize(pos, t):
    step = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    return {'duration_s': float(t[-1] - t[0]), 'path_m': float(step.sum()),
            'start_end_gap_m': float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
            'avg_speed_mps': float(step.sum() / (t[-1] - t[0]))}


def save_csv(path, t, pos):
    np.savetxt(path, np.c_[t, pos], delimiter=',', header='time,x,y,z', comments='', fmt='%.5f')


def plot(pos, info, png=None, show=True, raw_pos=None):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    if raw_pos is not None:
        ax.plot(raw_pos[:, 0], raw_pos[:, 1], '--', lw=1, color='gray', label='before loop closure')
    ax.plot(pos[:, 0], pos[:, 1], '-', lw=1.5)
    ax.plot(*pos[0, :2], 'go', ms=9, label='start')
    ax.plot(*pos[-1, :2], 'rs', ms=9, label='end')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    ax.set_title('RIDI path  |  %.1f m in %.0f s (%.2f m/s), end %.1f m from start' % (
        info['path_m'], info['duration_s'], info['avg_speed_mps'], info['start_end_gap_m']), fontsize=10)
    if raw_pos is not None:
        gap0 = np.linalg.norm(raw_pos[-1, :2] - raw_pos[0, :2])
        ax.set_title('RIDI path with loop closure  |  %.1f m in %.0f s\n(gap before closing: %.1f m)' % (
            info['path_m'], info['duration_s'], gap0), fontsize=10)
    ax.legend()
    if png:
        fig.savefig(png, dpi=120)
    if show:
        plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--placement', default='leg', choices=['leg', 'handheld', 'bag', 'body', 'auto'])
    ap.add_argument('--closed', action='store_true', help='walk ended where it started: close the loop')
    ap.add_argument('--no-show', action='store_true')
    a = ap.parse_args()
    data = dict(np.load(a.npz))
    model = ridi_model.SVRCascade()
    t0 = time.time()
    pos, tr = run(data, model, a.placement, verbose=True)
    raw_pos = None
    if a.closed:
        raw_pos, pos = pos, close_loop(pos)
    info = summarize(pos, data['t'])
    print(info, 'computed in %.1fs' % (time.time() - t0))
    base = a.npz.rsplit('.', 1)[0]
    save_csv(base + '_path.csv', data['t'], pos)
    plot(pos, info, base + '_path.png', show=not a.no_show, raw_pos=raw_pos)
