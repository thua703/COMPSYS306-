from pathlib import Path
from PIL import Image, ImageOps

# Change this path if your source folder has a different name.
SOURCE_DIR = Path(__file__).resolve().parent / 'processedImages'
OUTPUT_DIR = Path(__file__).resolve().parent / 'processedImages_resized'
TARGET_SIZE = (64, 48)  # PIL uses (width, height)
CLASSES = ('blue_light', 'road', 'sheep', 'speed', 'stop', 'yellow_light')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    if not SOURCE_DIR.is_dir():
        print(f'ERROR: Source folder not found: {SOURCE_DIR}')
        print('Edit SOURCE_DIR in this script to point to your image folder.')
        return

    if OUTPUT_DIR.resolve() == SOURCE_DIR.resolve():
        raise ValueError('Output and source folders must be different.')

    print(f'Source: {SOURCE_DIR}')
    print(f'Output: {OUTPUT_DIR}')
    print(f'Target: {TARGET_SIZE[0]} x {TARGET_SIZE[1]} RGB')
    print()
    total_ok = 0
    total_fail = 0

    for class_name in CLASSES:
        source_class = SOURCE_DIR / class_name
        output_class = OUTPUT_DIR / class_name
        if not source_class.is_dir():
            print(f'{class_name}: SKIPPED (folder not found)')
            continue

        paths = sorted(p for p in source_class.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        ok = 0
        fail = 0
        for src in paths:
            relative_path = src.relative_to(source_class)
            dst = output_class / relative_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(src) as img:
                    img = ImageOps.exif_transpose(img)
                    img = img.convert('RGB')
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    if dst.suffix.lower() in {'.jpg', '.jpeg'}:
                        img.save(dst, quality=95)
                    else:
                        img.save(dst)
                with Image.open(dst) as verify:
                    if verify.size != TARGET_SIZE or verify.mode != 'RGB':
                        raise ValueError('Output verification failed')
                ok += 1
            except Exception as exc:
                print(f'  ERROR: {src.name}: {exc}')
                fail += 1
        print(f'{class_name:13s} OK: {ok:4d} | Failed: {fail}')
        total_ok += ok
        total_fail += fail

    print(f'\nDONE: {total_ok} images converted; {total_fail} failed.')
    print(f'Resized images are in: {OUTPUT_DIR}')
    print('NOTE: If rerunning after changing labels, delete the old output folder first to avoid stale files.')


if __name__ == '__main__':
    main()
