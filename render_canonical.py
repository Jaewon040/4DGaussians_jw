# render_canonical.py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
import imageio

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_canonical_set(model_path, split, iteration, views, gaussians, pipeline, background, cam_type):
    """
    canonical Gaussians만 렌더링하여 저장
    
    Args:
        model_path: 모델 경로
        split: 'train' 또는 'test'
        iteration: 현재 iteration 번호
        views: 렌더링할 카메라 뷰
        gaussians: 가우시안 모델
        pipeline: 렌더링 파이프라인
        background: 배경색
        cam_type: 카메라 타입
    """
    # canonical 렌더링 결과 저장 경로 설정
    render_dir = os.path.join(model_path, "canonical", f"iter_{iteration}")
    render_path = os.path.join(render_dir, split, "renders")
    gts_path = os.path.join(render_dir, split, "gt")
    
    # 필요한 디렉토리 생성
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    # 렌더링 진행 정보 출력
    print(f"Rendering {len(views)} canonical Gaussians for {split} cameras...")
    
    # 결과 저장용 리스트
    render_images = []
    
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {split} canonical views")):
        # canonical 모드로 렌더링 (deformation 미적용)
        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type, canonical=True)["render"]
        
        # 렌더링 이미지 저장
        render_images.append(to8b(rendering).transpose(1, 2, 0))
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        
        # Ground truth 이미지 저장 (있는 경우)
        if cam_type != "PanopticSports":
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))
        else:
            gt = view['image'].cuda()
            torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))
    
    # 비디오 생성
    print(f"Creating {split} video from rendered images")
    video_path = os.path.join(render_dir, f"video_{split}.mp4")
    imageio.mimwrite(video_path, render_images, fps=30, quality=8)
    
    print(f"Canonical rendering for {split} complete!")

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, specific_camera=-1):
    """
    Train과 Test 카메라 세트에 대해 canonical Gaussians 렌더링
    
    Args:
        specific_camera: 특정 카메라만 렌더링할 경우 인덱스 지정 (-1이면 모든 카메라)
    """
    with torch.no_grad():
        # Gaussian 모델 로드
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type = scene.dataset_type
        
        # 배경색 설정
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 포인트 개수 출력
        print(f"Loaded model with {gaussians.get_xyz.shape[0]} points from iteration {iteration}")
        
        # Train 카메라에 대해 렌더링
        train_cameras = scene.getTrainCameras()
        if len(train_cameras) > 0:
            # 특정 카메라만 렌더링할 경우
            if specific_camera >= 0 and specific_camera < len(train_cameras):
                train_cameras = [train_cameras[specific_camera]]
                print(f"Rendering only train camera {specific_camera}")
            
            render_canonical_set(
                dataset.model_path, 
                "train", 
                iteration, 
                train_cameras, 
                gaussians, 
                pipeline, 
                background,
                cam_type
            )
        
        # Test 카메라에 대해 렌더링
        test_cameras = scene.getTestCameras()
        if len(test_cameras) > 0:
            # 특정 카메라만 렌더링할 경우
            if specific_camera >= 0 and specific_camera < len(test_cameras):
                test_cameras = [test_cameras[specific_camera]]
                print(f"Rendering only test camera {specific_camera}")
                
            render_canonical_set(
                dataset.model_path, 
                "test", 
                iteration, 
                test_cameras, 
                gaussians, 
                pipeline, 
                background,
                cam_type
            )
        
        # 비디오 카메라에 대해 렌더링 (옵션)
        if hasattr(dataset, 'render_video') and dataset.render_video:
            video_cameras = scene.getVideoCameras()
            if len(video_cameras) > 0:
                render_canonical_set(
                    dataset.model_path, 
                    "video", 
                    iteration, 
                    video_cameras, 
                    gaussians, 
                    pipeline, 
                    background,
                    cam_type
                )
        
        print("Canonical rendering complete!")

if __name__ == "__main__":
    # 명령줄 인수 파싱
    parser = ArgumentParser(description="Canonical Gaussians rendering script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load. -1 for latest.")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--configs", type=str, default="", help="Config file")
    parser.add_argument("--specific_camera", type=int, default=-1, help="Render only a specific camera. -1 for all cameras.")
    parser.add_argument("--render_video", action="store_true", help="Also render video cameras if available")
    
    args = get_combined_args(parser)
    print("Rendering canonical Gaussians from " + args.model_path)
    
    # 설정 파일이 제공된 경우 로드
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    # RNG 초기화
    safe_state(args.quiet)
    
    # 렌더링 실행
    render_sets(
        model.extract(args), 
        hyperparam.extract(args), 
        args.iteration, 
        pipeline.extract(args),
        args.specific_camera
    )
    
    print("Canonical rendering finished.")