import argparse
from pathlib import Path

import cv2
from tqdm import trange


def parser_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("video", type=Path)
    parser.add_argument("--save-dir", type=Path, default="./images")
    return parser.parse_args()


def main():
    args = parser_args()

    if str(args.video).endswith(".mp4"):
        videos = [args.video]
    else:
        videos = list(args.video.rglob("*.mp4"))

    for video in videos:
        video_name = video.stem
        camera_type = video.parent.name

        save_dir = args.save_dir / camera_type / video_name
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in trange(frame_count, desc=f"{camera_type}/{video_name}"):
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imwrite(str(save_dir / f"{i:06d}_{camera_type}.png"), frame)


if __name__ == "__main__":
    main()
