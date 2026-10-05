import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple annotation pipeline: extract dataset from video/images and review keypoints."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_extract = subparsers.add_parser(
        "extract", help="Extract frames and initialize dataset files without any AI inference."
    )
    parser_extract.add_argument("--input", required=True, help="Path to a video file or image folder.")
    parser_extract.add_argument("--output", default="outputs", help="Output root directory.")
    parser_extract.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Keep every Nth frame/image. Default: 1.",
    )

    parser_review = subparsers.add_parser("review", help="Open keypoint reviewer GUI.")
    parser_review.add_argument("scene_path", help="Path to generated scene folder (e.g., outputs/my_video).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        from src.extractor import run_extract

        run_extract(
            input_path=args.input,
            output_root=args.output,
            frame_step=args.frame_step,
        )
    elif args.command == "review":
        from src.reviewer import MatplotlibReviewer

        MatplotlibReviewer(args.scene_path)
    else:
        from src.gui import run_gui

        run_gui()


if __name__ == "__main__":
    main()
