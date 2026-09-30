"""Record a walk with the board, then run RIDI and show the path.

    python walk.py [--placement leg|handheld|bag|body|auto]

Press Enter to stop recording.
"""
import argparse
import datetime
import os
import threading

import numpy as np

import imu_to_ridi
import record_imu
import ridi_model
import run_ridi

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--placement', default='leg', choices=['leg', 'handheld', 'bag', 'body', 'auto'])
    ap.add_argument('--closed', action='store_true', help='walk ended where it started: close the loop')
    ap.add_argument('--use', help='skip recording and process this raw.csv instead')
    a = ap.parse_args()

    holder = {}
    def load():
        holder['model'] = ridi_model.SVRCascade()
    loader = threading.Thread(target=load, daemon=True)
    loader.start()

    if a.use:
        raw = a.use
    else:
        folder = os.path.join(HERE, '..', 'recordings', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(folder, exist_ok=True)
        raw = os.path.join(folder, 'raw.csv')
        ser = record_imu.open_board()
        print('Keep the board STILL for 2 seconds, then walk. Press Enter when done.')
        stop = threading.Event()
        record_imu.wait_for_enter(stop)
        try:
            record_imu.record(ser, raw, stop)
        except KeyboardInterrupt:
            stop.set()
        ser.close()
        print()

    data, info = imu_to_ridi.convert(raw)
    print('accel scale %.4f (%s), kept frames %d..%d of %d' % (
        info['accel_scale'], 'measured at rest' if info['scale_from_rest'] else 'DEFAULT, no still period found',
        info['trim'][0], info['trim'][1], info['frames_total']))
    if len(data['t']) < 1000:
        raise SystemExit('Walk was too short (need at least ~5 s of walking).')

    print('Loading RIDI model...')
    loader.join()
    pos, _ = run_ridi.run(data, holder['model'], a.placement)
    raw_pos = None
    if a.closed:
        raw_pos, pos = pos, run_ridi.close_loop(pos)
    summary = run_ridi.summarize(pos, data['t'])
    print(summary)
    base = raw.rsplit('.', 1)[0]
    np.savez(base + '_ridi.npz', **data)
    run_ridi.save_csv(base + '_path.csv', data['t'], pos)
    run_ridi.plot(pos, summary, base + '_path.png', raw_pos=raw_pos)


if __name__ == '__main__':
    main()
