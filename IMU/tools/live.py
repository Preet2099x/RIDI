"""Shows the RIDI path in real time while you walk.

    python live.py [--placement leg|handheld|bag|body|auto]

Hold the board still for the first 2 seconds (accelerometer calibration), then walk.
Close the window to stop. The live path uses RIDI's running bias correction only;
walk.py gives the final, globally optimized path afterwards.
"""
import argparse
import queue
import threading
import time

import numpy as np

import imu_to_ridi as conv
import record_imu
import ridi_core as rc
import ridi_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--placement', default='leg', choices=['leg', 'handheld', 'bag', 'body', 'auto'])
    a = ap.parse_args()

    print('Loading RIDI model...')
    model = ridi_model.SVRCascade()
    ser = record_imu.open_board()
    q_in = queue.Queue()
    stop = threading.Event()
    state = {'pos': np.zeros((1, 3)), 'status': 'Hold the board still for 2 s...'}
    lock = threading.Lock()

    def reader():
        buf = b''
        last_s = 0.0
        got = False
        while not stop.is_set():
            if not got and time.time() - last_s > 1.0:
                last_s = time.time()
                ser.write(b'S')
            chunk = ser.read(4096)
            buf += chunk
            *lines, buf = buf.split(b'\n')
            for raw in lines:
                row = record_imu.parse(raw.decode('ascii', 'ignore'))
                if row:
                    got = True
                    q_in.put(row)

    def worker():
        label = None if a.placement == 'auto' else model.class_id(a.placement)
        tr = rc.Trajectory(model, force_label=label)
        prev = None
        t_next = None
        calib = []
        scale = None
        t0 = None
        last_pub = 0.0
        wrap = 0
        last_us = None
        while not stop.is_set():
            try:
                row = q_in.get(timeout=0.2)
            except queue.Empty:
                continue
            us = row[0]
            if last_us is not None and us < last_us - 2 ** 31:
                wrap += 1
            last_us = us
            t = (us + wrap * 2.0 ** 32) * 1e-6
            cur = np.array([t] + row[1:11])
            if prev is not None and cur[7:11] @ prev[7:11] < 0:
                cur[7:11] *= -1
            if prev is None:
                prev = cur
                t0 = t
                t_next = t
                continue
            # emit uniform 200 Hz samples between prev and cur
            while t_next <= cur[0]:
                f = (t_next - prev[0]) / (cur[0] - prev[0])
                s = prev + f * (cur - prev)
                s[7:11] /= np.linalg.norm(s[7:11])
                t_next += 1.0 / conv.RATE
                if scale is None:
                    calib.append(s)
                    if s[0] - t0 >= 2.0:
                        c = np.array(calib)
                        scale, ok = conv.estimate_accel_scale(c[:, 1:4], c[:, 4:7])
                        with lock:
                            state['status'] = ('accel scale %.4f. Walk!' % scale) if ok else \
                                'No still period found, using default scale. Walk!'
                    continue
                gyro, lin, grav = conv.derive(s[None, 1:4], s[None, 4:7], s[None, 7:11], scale)
                tr.add_record(s[0] - t0, gyro[0], lin[0], grav[0], s[7:11])
                if time.time() - last_pub > 0.2:
                    last_pub = time.time()
                    with lock:
                        state['pos'] = tr.positions()
                        state['n'] = tr.n

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    line, = ax.plot([], [], '-', lw=1.5)
    head, = ax.plot([], [], 'ro')
    ax.plot(0, 0, 'go', ms=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    title = ax.set_title('')
    try:
        while plt.fignum_exists(fig.number):
            with lock:
                pos = state['pos'].copy()
                status = state['status']
            if len(pos) > 1:
                line.set_data(pos[:, 0], pos[:, 1])
                head.set_data([pos[-1, 0]], [pos[-1, 1]])
                lo = np.minimum(pos[:, :2].min(0), -1) - 1
                hi = np.maximum(pos[:, :2].max(0), 1) + 1
                span = (hi - lo).max()
                mid = (hi + lo) / 2
                ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
                ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
            step = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum() if len(pos) > 1 else 0
            title.set_text('%s   |   %.1f m walked' % (status, step))
            plt.pause(0.2)
    finally:
        stop.set()
        try:
            ser.write(b'P')
            ser.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
