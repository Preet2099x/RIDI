"""Records the board's 200 Hz stream ('D,...' lines) to a CSV file.

    python record_imu.py [out.csv] [--port COM9]

Stops on Enter or Ctrl+C.
"""
import argparse
import os
import sys
import threading
import time

import serial
import serial.tools.list_ports

TEENSY_VID = 0x16C0
COLUMNS = 't_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,q0,q1,q2,q3,overruns'


def find_port():
    for p in serial.tools.list_ports.comports():
        if p.vid == TEENSY_VID:
            return p.device
    return None


def open_board(port=None):
    port = port or find_port()
    if port is None:
        sys.exit('Board not found. Plug in the Teensy and close other programs using it (serial monitor).')
    ser = serial.Serial(port, 115200, timeout=0.2)
    ser.reset_input_buffer()
    ser.write(b'S')          # start streaming (also skips the mag calibration at boot)
    return ser


def parse(line):
    """'D,...' text line -> list of 12 floats, or None."""
    if not line.startswith('D,'):
        return None
    p = line.strip().split(',')
    if len(p) != 13:
        return None
    try:
        return [float(x) for x in p[1:]]
    except ValueError:
        return None


def record(ser, out_path, stop_event, quiet=False):
    """Writes lines until stop_event is set. Returns number of samples."""
    n = 0
    last_print = time.time()
    last_s = 0.0
    with open(out_path, 'w') as f:
        f.write(COLUMNS + '\n')
        buf = b''
        while not stop_event.is_set():
            try:
                chunk = ser.read(4096)
            except serial.SerialException:
                print('Board disconnected.')
                break
            if n == 0 and time.time() - last_s > 1.0:
                # the board may still be starting up and swallow the first 'S'
                last_s = time.time()
                ser.write(b'S')
            if not chunk:
                continue
            buf += chunk
            *lines, buf = buf.split(b'\n')
            for raw in lines:
                row = parse(raw.decode('ascii', 'ignore'))
                if row is None:
                    continue
                f.write('%d,%s\n' % (row[0], ','.join('%.6g' % v for v in row[1:])))
                n += 1
            if not quiet and time.time() - last_print > 1.0:
                last_print = time.time()
                print('\r  recording... %d samples (%.0f s)   ' % (n, n / 166.0), end='', flush=True)
    try:
        ser.write(b'P')
    except serial.SerialException:
        pass
    return n


def wait_for_enter(stop_event):
    def w():
        try:
            input()
        except EOFError:
            pass
        stop_event.set()
    threading.Thread(target=w, daemon=True).start()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='raw.csv')
    ap.add_argument('--port')
    a = ap.parse_args()
    ser = open_board(a.port)
    stop = threading.Event()
    wait_for_enter(stop)
    print('Recording. Press Enter to stop.')
    try:
        n = record(ser, a.out, stop)
    except KeyboardInterrupt:
        stop.set()
        n = -1
    ser.close()
    print('\nSaved', os.path.abspath(a.out), n)
