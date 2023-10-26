from scripts.extract_clip_gs import run as run_base
from scripts.extract_frame_clip_gs_shape import run as run_movieqa_1
from scripts.get_video_shape_gs import run as run_movieqa_2


@dataclass
class Config:
    bucket_name: str = 'gs://ai2-yj'
    dataset: str = 'pororoqa'

    clip_model_type: str = 'ViT-B/32'


options = {
    'pororoqa': {
        'blob_dir': 'PororoQA/Scenes_Dialogues',
        'out_blob_dir': 'PororoQA/frame_clip'
        'chunk_size': 20,
        'depth_dir': 3,
        'filetype': '.gif',
    },
    'dramaqa': {
        'blob_dir': 'DramaQA/AnotherMissOh_images',
        'out_blob_dir': 'DramaQA/frame_clip'
        'chunk_size': 20,
        'depth_dir': 4,
        'filetype': '.jpg',
    },
}


def main():
    args = get_args()
    if args.dataset == 'movieqa':
        run_movieqa_1(args)
        run_movieqa_2(args)
    else:
        for k, v in options[args.dataset]:
            setattr(args, k, v)
        run_base(args)


if __name__ == '__main__':
    main()
