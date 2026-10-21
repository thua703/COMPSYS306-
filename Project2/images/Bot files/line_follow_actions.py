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
FAST_STOP_LABELS = {'blue_light', 'sheep'}
FAST_STOP_FRAMES = 2
PRINT_EVERY_N_FRAMES = 1

SAVE_DIR = Path('inference_captures')
SAVE_INTERVAL = 0.5  # Minimum seconds between saves for each raw prediction class
STOP_SECONDS = 5.0
SPEED_SECONDS = 2.0


class ActionController:
    """Non-blocking actions; sheep/blue/STOP override high-speed driving."""

    def __init__(self):
        self.blue_hold = False
        self.sheep_hold = False
        self.stop_until = 0.0
        self.speed_until = 0.0
        self.motion = None

    def confirm(self, label, now):
        if label == 'stop':
            self.stop_until = now + STOP_SECONDS
            self.speed_until = 0.0
        elif label == 'speed':
            if not (self.blue_hold or self.sheep_hold or now < self.stop_until):
                self.speed_until = now + SPEED_SECONDS
        elif label == 'blue_light':
            self.blue_hold = True
            self.speed_until = 0.0
        elif label == 'yellow_light':
            self.blue_hold = False
        elif label == 'sheep':
            self.sheep_hold = True
            self.speed_until = 0.0

    def update(self, now):
        blocked = self.blue_hold or self.sheep_hold or now < self.stop_until
        desired = 'STOPPED' if blocked else ('HIGH' if now < self.speed_until else 'LOW')
        if desired != self.motion:
            if desired == 'STOPPED':
                pibot.lfStop()
                pibot.robot.stop()
            else:
                if desired == 'HIGH':
                    pibot.lfSpeedHigh()
                else:
                    pibot.lfSpeedLow()
                if self.motion != 'HIGH' and self.motion != 'LOW':
                    pibot.lfStart()
            self.motion = desired
            print(f'  [ACTION] {desired}', flush=True)
        return desired


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
    latched = set()
    actions = ActionController()
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
        actions.motion = 'LOW'
        print('Line following + filtered SVM inference. Press x to stop (no Enter needed).')

        while True:
            # Check for x before each camera frame.
            if terminal_fd is not None:
                ready, _, _ = select.select([terminal_fd], [], [], 0)
                if ready and sys.stdin.read(1).lower() == 'x':
                    print('x received. Stopping robot...')
                    break

            actions.update(time.monotonic())
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

            # ============================================================
            # SHEEP WAITING MODE
            # ============================================================

            if actions.sheep_hold:

                # Ignore all signs while waiting for the sheep to disappear.
                # Only consecutive road predictions can release the sheep hold.

                if raw == 'road':
                    road_streak += 1
                else:
                    road_streak = 0

                if road_streak >= ROAD_RELEASE_FRAMES:

                    # Sheep is considered removed.
                    actions.sheep_hold = False

                    confirmed = None
                    road_streak = 0

                    # Clear old predictions and confirmation locks.
                    history.clear()
                    latched.clear()

                    event = 'SHEEP REMOVED'

            # ============================================================
            # NORMAL MODE
            # ============================================================

            else:

                # Update consecutive road counter.
                if raw == 'road':
                    road_streak += 1
                else:
                    road_streak = 0

                # Re-arm labels that have left the voting window.
                latched.intersection_update(set(history))

                if road_streak >= ROAD_RELEASE_FRAMES:
                    confirmed = None
                    latched.clear()

                # --------------------------------------------------------
                # Fast confirmation: Blue light and Sheep
                # --------------------------------------------------------

                fast_candidate = None

                if len(history) >= FAST_STOP_FRAMES:

                    recent = list(history)[-FAST_STOP_FRAMES:]

                    if all(label == recent[0] for label in recent):

                        if recent[0] in FAST_STOP_LABELS:
                            fast_candidate = recent[0]

                if fast_candidate is not None and fast_candidate not in latched:

                    confirmed = fast_candidate
                    latched.add(fast_candidate)

                    actions.confirm(fast_candidate, time.monotonic())

                    event = f'FAST CONFIRMED {fast_candidate}'

                    # Reset road counter when entering Sheep Waiting Mode.
                    if fast_candidate == 'sheep':
                        road_streak = 0

                # --------------------------------------------------------
                # Normal confirmation: STOP / SPEED / Yellow
                # --------------------------------------------------------

                else:

                    counts = Counter(
                        label for label in history
                        if label != 'road' and label not in FAST_STOP_LABELS
                    )

                    if counts:

                        candidate, votes = counts.most_common(1)[0]

                        if votes >= VOTE_THRESHOLD and candidate not in latched:

                            confirmed = candidate
                            latched.add(candidate)

                            actions.confirm(candidate, time.monotonic())

                            event = f'CONFIRMED {candidate}'

            motion = actions.update(time.monotonic())

            state = 'LOCKED' if confirmed is not None else 'SEARCHING'
            confirmed_text = confirmed if confirmed is not None else 'none'

            if frame_count % PRINT_EVERY_N_FRAMES == 0 or event:
                print(
                    f'Frame {frame_count:05d} | Raw: {raw:<13} | '
                    f'Confirmed: {confirmed_text:<13} | '
                    f'State: {state:<9} | '
                    f'Road streak: {road_streak} | '
                    f'Motion: {motion} | Blue: {actions.blue_hold} | Sheep: {actions.sheep_hold} | '
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
