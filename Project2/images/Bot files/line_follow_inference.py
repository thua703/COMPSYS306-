import time
import sys
import select
import termios
import tty
import pickle
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pibot

MODEL_PATH = Path('svm_final_model_V6.pkl')
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48
VOTE_WINDOW = 7
VOTE_THRESHOLD = 5
ROAD_RELEASE_FRAMES = 7
PRINT_EVERY_N_FRAMES = 1

SAVE_DIR = Path('inference_captures')
SAVE_INTERVAL = 0.5  # Minimum seconds between saves for each raw prediction class


def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS
    )
    return (np.asarray(image, dtype=np.float32) / 255.0).reshape(1, -1)


def main():
    print('Loading model:', MODEL_PATH)
    with MODEL_PATH.open('rb') as f:
        model = pickle.load(f)
    print('Classes:', list(model.classes_))

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print('Saving non-road frames to:', SAVE_DIR.resolve())

    history = deque(maxlen=VOTE_WINDOW)
    confirmed = None
    road_streak = 0
    last_saved_time = {}
    camera_initialized = False
    line_initialized = False
    frame_count = 0
    terminal_fd = None
    terminal_settings = None

    try:
        # SSH/Linux terminal: read x immediately, without pressing Enter.
        if sys.stdin.isatty():
            terminal_fd = sys.stdin.fileno()
            terminal_settings = termios.tcgetattr(terminal_fd)
            tty.setcbreak(terminal_fd)
        else:
            print('Warning: stdin is not a terminal; x-key exit is unavailable.')

        pibot.camInit()
        camera_initialized = True
        pibot.lfInit()
        line_initialized = True
        pibot.lfSpeedLow()
        print('Starting line follower in 3 seconds...')
        time.sleep(3)
        pibot.lfStart()
        print('Line following + filtered SVM inference. Press x to stop (no Enter needed).')

        while True:
            # Check for x before each camera frame.
            if terminal_fd is not None:
                ready, _, _ = select.select([terminal_fd], [], [], 0)
                if ready and sys.stdin.read(1).lower() == 'x':
                    print('x received. Stopping robot...')
                    break

            frame = pibot.camWait()
            if frame is None:
                continue

            frame_count += 1
            features = preprocess_frame(frame)
            start = time.perf_counter()
            raw = str(model.predict(features)[0])
            inference_ms = (time.perf_counter() - start) * 1000.0

            history.append(raw)
            event = ''

            if confirmed is None:
                counts = Counter(label for label in history if label != 'road')
                if counts:
                    candidate, votes = counts.most_common(1)[0]
                    if votes >= VOTE_THRESHOLD:
                        confirmed = candidate
                        road_streak = 0
                        event = f'CONFIRMED {confirmed}'
            else:
                if raw == 'road':
                    road_streak += 1
                else:
                    road_streak = 0

                if road_streak >= ROAD_RELEASE_FRAMES:
                    event = f'RELEASED {confirmed}'
                    confirmed = None
                    history.clear()
                    road_streak = 0

            state = 'LOCKED' if confirmed is not None else 'SEARCHING'
            confirmed_text = confirmed if confirmed is not None else 'none'

            if frame_count % PRINT_EVERY_N_FRAMES == 0 or event:
                print(
                    f'Frame {frame_count:05d} | Raw: {raw:<13} | '
                    f'Confirmed: {confirmed_text:<13} | '
                    f'State: {state:<9} | '
                    f'Road streak: {road_streak} | '
                    f'Inference: {inference_ms:.1f} ms'
                    + (f' | {event}' if event else ''),
                    flush=True,
                )

            # Save camera image for later inspection, based on RAW prediction.
            if raw != 'road':
                now = time.monotonic()
                last = last_saved_time.get(raw, float('-inf'))
                if now - last >= SAVE_INTERVAL:
                    image_path = SAVE_DIR / f'frame_{frame_count:05d}_{raw}.jpg'
                    if cv2.imwrite(str(image_path), frame):
                        last_saved_time[raw] = now
                        print(f'  [SAVED] {image_path}', flush=True)
                    else:
                        print(f'  [WARNING] Could not save {image_path}', flush=True)

    except KeyboardInterrupt:
        print('\nCtrl+C received.')
    finally:
        # Stop motors FIRST, then clean up camera/line follower.
        if line_initialized:
            try:
                pibot.lfStop()
            except Exception as e:
                print('lfStop error:', e)
        try:
            pibot.robot.stop()
        except Exception as e:
            print('robot.stop error:', e)

        if line_initialized:
            try:
                pibot.lfStop()
            except Exception as e:
                print('lfStop error:', e)
            try:
                pibot.lfDeinit()
            except Exception as e:
                print('lfDeinit error:', e)
        try:
            pibot.robot.stop()
        except Exception as e:
            print('second robot.stop error:', e)
        if camera_initialized:
            try:
                pibot.camDeinit()
            except Exception as e:
                print('camDeinit error:', e)
        if terminal_fd is not None and terminal_settings is not None:
            termios.tcsetattr(terminal_fd, termios.TCSADRAIN, terminal_settings)
        print('Finished.')


if __name__ == '__main__':
    main()
