import os
import json
from PIL import Image

from model.text2room_pipeline import Text2RoomPipeline
from model.utils.opt import get_default_parser
from model.utils.utils import save_poisson_mesh, generate_first_image

import torch


@torch.no_grad()
def main(args):
    # load trajectories
    trajectories = json.load(open(args.trajectory_file, "r"))

    # check if there is a custom prompt in the first trajectory
    # would use it to generate start image, if we have to
    if "prompt" in trajectories[0]:
        args.prompt = trajectories[0]["prompt"]

    # get first image from text prompt or saved image folder
    # 这段代码用于获取第一张图像，它可以是根据文本提示生成的图像，也可以是从保存的图像文件夹中读取的图像。
    if (not args.input_image_path) or (not os.path.isfile(args.input_image_path)):
        first_image_pil = generate_first_image(args)
    else:
        first_image_pil = Image.open(args.input_image_path)

    # load pipeline
    pipeline = Text2RoomPipeline(args, first_image_pil=first_image_pil)

    # generate using all trajectories
    offset = 1  # have the start image already
    # 使用所有轨迹生成图像：对于每个预定义的轨迹，将其设置为当前轨迹，并调用 pipeline.generate_images() 方法生成相应的图像序列。offset 变量用于记录当前生成的图像数量
    for t in trajectories:
        pipeline.set_trajectory(t)
        offset = pipeline.generate_images(offset=offset)

    # save outputs before completion
    pipeline.clean_mesh()
    intermediate_mesh_path = pipeline.save_mesh("after_generation.ply")
    
    # 用于从一组点云数据和法向量生成平滑的 3D 网格。
    save_poisson_mesh(intermediate_mesh_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)

    # -------------------------------------------------------------------
    # run completion
    # 对生成的场景进行补全，以填充可能的空洞和不完整区域。
    pipeline.args.update_mask_after_improvement = True
    pipeline.complete_mesh(offset=offset)
    pipeline.clean_mesh()

    # Now no longer need the models
    pipeline.remove_models()

    # save outputs after completion
    final_mesh_path = pipeline.save_mesh()

    # run poisson mesh reconstruction
    mesh_poisson_path = save_poisson_mesh(final_mesh_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)

    # save additional output
    pipeline.save_animations()
    pipeline.load_mesh(mesh_poisson_path)
    pipeline.save_seen_trajectory_renderings(apply_noise=False, add_to_nerf_images=True)
    pipeline.save_nerf_transforms()
    pipeline.save_seen_trajectory_renderings(apply_noise=True)

    print("Finished. Outputs stored in:", args.out_path)


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(args)
