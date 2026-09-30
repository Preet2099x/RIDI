"""Turns the board's raw stream into the signals RIDI wants.

    python imu_to_ridi.py raw.csv [out.npz]

RIDI wants (Android convention): 200 Hz, gyro in rad/s, linear acceleration in
m/s^2 (gravity removed), the gravity vector (points up, in the sensor frame)
and the sensor->world orientation quaternion.
"""
import sys

import numpy as np

G = 9.80665
RATE = 200.0


def read_raw(path):
    d = np.loadtxt(path, delimiter=',', skiprows=1)
    t = d[:, 0].copy()
    # micros() is a 32-bit counter, unwrap it
    wraps = np.cumsum(np.r_[0, np.diff(t) < -2 ** 31])
    t = (t + wraps * 2.0 ** 32) * 1e-6
    d[:, 0] = t - t[0]
    keep = np.r_[True, np.diff(d[:, 0]) > 0]
    return d[keep]


def resample(d, rate=RATE):
    """Interpolate every column onto a uniform time grid."""
    t = d[:, 0]
    grid = np.arange(0.0, t[-1], 1.0 / rate)
    out = np.zeros((len(grid), d.shape[1]))
    out[:, 0] = grid
    q = d[:, 7:11].copy()
    for i in range(1, len(q)):                      # keep the quaternion on one side
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    d = d.copy()
    d[:, 7:11] = q
    for c in range(1, d.shape[1]):
        out[:, c] = np.interp(grid, t, d[:, c])
    out[:, 7:11] /= np.linalg.norm(out[:, 7:11], axis=1, keepdims=True)
    return out


def quat_third_row(q):
    """Third row of the sensor->world rotation = world 'up' in the sensor frame."""
    w, x, y, z = q.T
    return np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)


def still_mask(a_g, gyro_dps, rate=RATE):
    k = int(rate)
    def mean(x):
        return np.convolve(x, np.ones(k) / k, mode='same')
    rate_dps = mean(np.linalg.norm(gyro_dps, axis=1))
    an = np.linalg.norm(a_g, axis=1)
    dev = mean(np.abs(an - mean(an)))
    return (rate_dps < 1.5) & (dev < 0.01) & (np.abs(an - 1) < 0.1)


def estimate_accel_scale(a_g, gyro_dps):
    """This board's accelerometer reads a little off 1 g at rest; find the factor."""
    m = still_mask(a_g, gyro_dps)
    if m.sum() < RATE:
        return 1.0, False
    s = 1.0 / np.median(np.linalg.norm(a_g[m], axis=1))
    return float(np.clip(s, 0.9, 1.1)), True


def derive(a_g, gyro_dps, q, scale):
    """Raw board signals -> RIDI signals (works on (n,3)/(n,4) arrays)."""
    up = quat_third_row(q)
    grav = up * G
    lin = a_g * scale * G - grav
    return gyro_dps * (np.pi / 180.0), lin, grav


def trim_still(lin, gyro, margin_s=1.0):
    """Cut the still stretches before walking starts / after it ends."""
    k = int(RATE)
    act = np.convolve(np.linalg.norm(lin, axis=1), np.ones(k) / k, mode='same')
    moving = np.where(act > 0.5)[0]
    if len(moving) == 0:
        return 0, len(lin)
    m = int(margin_s * RATE)
    return max(moving[0] - m, 0), min(moving[-1] + m, len(lin))


def convert(raw_path, trim=True):
    d = resample(read_raw(raw_path))
    a_g, g_dps, q = d[:, 1:4], d[:, 4:7], d[:, 7:11]
    scale, found = estimate_accel_scale(a_g, g_dps)
    gyro, lin, grav = derive(a_g, g_dps, q, scale)
    s, e = trim_still(lin, gyro) if trim else (0, len(d))
    info = {'accel_scale': scale, 'scale_from_rest': found, 'trim': (s, e), 'frames_total': len(d)}
    return {'t': d[s:e, 0] - d[s, 0], 'gyro': gyro[s:e], 'lin': lin[s:e], 'grav': grav[s:e], 'quat': q[s:e]}, info


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data, info = convert(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].rsplit('.', 1)[0] + '_ridi.npz'
    np.savez(out, **data)
    print(info)
    print('wrote', out, len(data['t']), 'frames')
