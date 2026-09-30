# IMU
## Running RIDI on this board

1. Plug the board in (COM port is found automatically). Firmware with the 200 Hz
   stream mode is already flashed (`pio run -t upload` to re-flash).
2. Double-click `walk.bat`: keep the board still 2 s, walk, press Enter. The path
   is shown and saved in `recordings/<time>/`.
   `live.bat` shows the path while you walk instead.
3. Options: `--placement leg|handheld|bag|body|auto` (default `leg`; the pretrained
   classifier is unreliable on this board, so `auto` is not recommended).

The RIDI algorithm is ported to Python in `tools/ridi_core.py` (checked against the
original C++ result: same path within about 5 mm on average). The first run converts the
2 GB pretrained model to a small cache in `tools/model_cache/` (one time, about 3 minutes).

For walks that end where they started, use `walk_loop.bat` instead of `walk.bat`. It closes
the start/end gap by spreading it over the path (the uncorrected path is drawn dashed).
