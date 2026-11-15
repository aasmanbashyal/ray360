"""Use an application-owned decoder with ray360's geometry and shared map cache.

uv run --with av python examples/video_frames.py stitched.mp4

This sequential example prints one view's dimensions and original presentation
time. Decode, seek, timestamp policy, audio and encoding remain the application's job.
"""

import argparse
import json
from pathlib import Path

from ray360 import MapCache, Panorama, Perspective


def iter_panoramas(path):
    """Decode stored RGB pixels in presentation order, retaining timestamps in seconds."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("The input has no video stream.")
        for frame in container.decode(video=0):
            timestamp = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None and frame.time_base is not None
                else None
            )
            yield Panorama(
                frame.to_ndarray(format="rgb24"), source_id=str(path), timestamp=timestamp
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--json", type=Path, help="Write frame timestamps and view shapes")
    args = parser.parse_args()
    target, cache = Perspective(320, 180), MapCache()
    # Opening exclusively before decoding avoids overwriting a previous run.
    from contextlib import nullcontext

    with args.json.open("x", encoding="utf-8") if args.json else nullcontext() as stream:
        if stream:
            stream.write("[\n")
        for index, panorama in enumerate(iter_panoramas(args.video)):
            view = panorama.view(target, cache=cache)
            print(view.timestamp, view.image.shape)
            if stream:
                if index:
                    stream.write(",\n")
                json.dump(
                    {"index": index, "timestamp": view.timestamp, "shape": view.image.shape}, stream
                )
        if stream:
            stream.write("\n]\n")


if __name__ == "__main__":
    main()
