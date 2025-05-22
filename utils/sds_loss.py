import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List, Union, Tuple
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm import tqdm

class StableDiffusionSDS(nn.Module):
    """
    Score Distillation Sampling Loss 구현, Stable Diffusion 모델 사용
    """
    def __init__(
        self,
        sd_version='runwayml/stable-diffusion-v1-5',
        device="cuda",
        guidance_scale=7.5,
        min_step_percent=0.02,
        max_step_percent=0.98,
    ):
        super().__init__()
        self.device = device
        self.guidance_scale = guidance_scale
        self.min_step_percent = min_step_percent
        self.max_step_percent = max_step_percent
        
        # Stable Diffusion 파이프라인 로드
        print(f"Loading Stable Diffusion {sd_version}...")
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            sd_version,
            torch_dtype=torch.float16,
        ).to(device)
        
        # DDIM 스케줄러 설정
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
        
        # 메모리 절약을 위해 VAE는 CPU에 보관, 필요할 때만 GPU로 이동
        self.sd_pipeline.vae.to("cpu")
        
        # UNet과 Text Encoder만 GPU에 유지
        self.sd_pipeline.unet.to(device)
        self.sd_pipeline.text_encoder.to(device)
        
        # 메모리 최적화
        self.sd_pipeline.unet.eval()
        self.sd_pipeline.text_encoder.eval()
        self.sd_pipeline.vae.eval()
        
        # 실제 사용할 타임스텝 범위 계산
        self.num_train_timesteps = self.sd_pipeline.scheduler.config.num_train_timesteps
        
    def get_text_embeddings(self, prompt: Union[str, List[str]], negative_prompt: str = "") -> torch.Tensor:
        """
        프롬프트에서 텍스트 임베딩 생성
        
        Args:
            prompt: 텍스트 프롬프트 (단일 문자열 또는 문자열 리스트)
            negative_prompt: 부정적인 프롬프트 (없으면 빈 문자열)
            
        Returns:
            텍스트 임베딩
        """
        # 배치 처리를 위해 단일 프롬프트를 리스트로 변환
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # 긍정적 프롬프트 임베딩
        text_input = self.sd_pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.sd_pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_input.input_ids.to(self.device)
        with torch.no_grad():
            positive_embeddings = self.sd_pipeline.text_encoder(text_input_ids)[0]
            
        # 부정적 프롬프트 임베딩 (있는 경우)
        if negative_prompt:
            negative_input = self.sd_pipeline.tokenizer(
                [negative_prompt] * len(prompt),
                padding="max_length",
                max_length=self.sd_pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            negative_input_ids = negative_input.input_ids.to(self.device)
            with torch.no_grad():
                negative_embeddings = self.sd_pipeline.text_encoder(negative_input_ids)[0]
                
            # Classifier-free guidance를 위해 negative와 positive 임베딩 결합
            text_embeddings = torch.cat([negative_embeddings, positive_embeddings])
        else:
            # 부정적 프롬프트가 없는 경우 동일한 임베딩을 사용하여 uncond 생성
            text_embeddings = torch.cat([positive_embeddings, positive_embeddings])
            
        return text_embeddings
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        이미지를 VAE 잠재 공간으로 인코딩
        
        Args:
            images: [B, 3, H, W] 범위 [-1, 1]의 이미지 텐서
            
        Returns:
            잠재 표현 (gradient 유지, 올바른 장치에 위치)
        """
        # CRITICAL: Input gradient 확인
        if not images.requires_grad:
            print(f"ERROR: Input images to encode_images do not require grad!")
        
        # 입력 이미지가 어느 장치에 있는지 확인
        input_device = images.device
        print(f"[DEVICE DEBUG] Input images on device: {input_device}")
        
        # VAE를 입력과 같은 장치로 이동
        print(f"[DEVICE DEBUG] Moving VAE to {input_device}")
        self.sd_pipeline.vae = self.sd_pipeline.vae.to(input_device)
        
        # 배치 처리
        latents_list = []
        batch_size = images.shape[0]
        
        # 메모리 제한을 위해 작은 배치로 처리
        sub_batch_size = 4
        for i in range(0, batch_size, sub_batch_size):
            sub_batch = images[i:i+sub_batch_size]
            
            # 서브배치가 올바른 장치에 있는지 확인
            print(f"[DEVICE DEBUG] sub_batch device: {sub_batch.device}")
            
            # VAE 인코딩 - gradient 유지하면서 같은 장치에서 처리
            latents = self.sd_pipeline.vae.encode(sub_batch.to(torch.float16)).latent_dist.sample()
            latents = 0.18215 * latents  # SD 모델을 위한 스케일링
            
            # 결과가 올바른 장치에 있는지 확인
            print(f"[DEVICE DEBUG] latents device after encoding: {latents.device}")
            
            # 입력과 같은 장치로 명시적으로 이동 (혹시 모를 상황 대비)
            latents = latents.to(input_device)
            
            # Gradient 확인
            if not latents.requires_grad:
                print(f"WARNING: latents lost gradient after VAE encoding!")
            
            latents_list.append(latents)
        
        # 결과 텐서 생성        
        result = torch.cat(latents_list, dim=0)
        
        # 최종 결과가 올바른 장치에 있는지 확인
        print(f"[DEVICE DEBUG] Final result device: {result.device}")
        
        # VAE를 CPU로 이동하지 말고 그대로 두기 (메모리 문제가 있다면 나중에 조정)
        # self.sd_pipeline.vae = self.sd_pipeline.vae.to("cpu")  # 이 줄을 주석 처리
        
        # Final gradient check
        if not result.requires_grad:
            print(f"ERROR: Final latents do not require grad!")
        
        return result
    
    
    def forward(
        self,
        rendered_images: torch.Tensor,
        prompt: Union[str, List[str]],
        negative_prompt: str = "",
        guidance_scale: Optional[float] = None,
        t_range: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        batch_size = rendered_images.shape[0]
        
        # 입력 장치 확인
        input_device = rendered_images.device
        print(f"[DEVICE DEBUG] Input rendered_images device: {input_device}")
        
        # CRITICAL: rendered_images가 gradient를 가지는지 확인
        if not rendered_images.requires_grad:
            print(f"CRITICAL ERROR: Input rendered_images does not require grad!")
            return torch.tensor(0.0, requires_grad=True, device=input_device)
        
        # Ensure images are in the correct range
        if rendered_images.min() < 0 or rendered_images.max() > 1:
            rendered_images = torch.clamp(rendered_images, 0.0, 1.0)
        
        # Convert to [-1, 1] range for SD model
        rendered_images_scaled = 2 * rendered_images - 1
        rendered_images_fp16 = rendered_images_scaled.to(torch.float16)
        
        # 장치 확인
        print(f"[DEVICE DEBUG] rendered_images_fp16 device: {rendered_images_fp16.device}")
        
        # Process prompts
        prompt_list = [prompt] * batch_size if isinstance(prompt, str) else prompt
        neg_prompt_list = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
        
        # 모든 SD 모델 컴포넌트가 같은 장치에 있는지 확인
        print(f"[DEVICE DEBUG] UNet device: {next(self.sd_pipeline.unet.parameters()).device}")
        print(f"[DEVICE DEBUG] Text encoder device: {next(self.sd_pipeline.text_encoder.parameters()).device}")
        
        # 필요하다면 모델들을 입력과 같은 장치로 이동
        if next(self.sd_pipeline.unet.parameters()).device != input_device:
            print(f"[DEVICE DEBUG] Moving UNet to {input_device}")
            self.sd_pipeline.unet = self.sd_pipeline.unet.to(input_device)
        
        if next(self.sd_pipeline.text_encoder.parameters()).device != input_device:
            print(f"[DEVICE DEBUG] Moving text encoder to {input_device}")
            self.sd_pipeline.text_encoder = self.sd_pipeline.text_encoder.to(input_device)
        
        # Encode images
        image_latents = self.encode_images(rendered_images_fp16)
        
        if not image_latents.requires_grad:
            print(f"ERROR: image_latents lost gradient after encoding!")
            return torch.tensor(0.0, requires_grad=True, device=input_device)
        
        print(f"[DEVICE DEBUG] image_latents device: {image_latents.device}")
        
        # Get text embeddings for all prompts
        text_embeddings_list = []
        for i in range(batch_size):
            text_emb = self.get_text_embeddings(prompt_list[i], neg_prompt_list[i])
            # 텍스트 임베딩도 올바른 장치에 있는지 확인
            text_emb = text_emb.to(input_device)
            text_embeddings_list.append(text_emb)
        
        # Sample timesteps
        min_step = int((t_range[0] if t_range else self.min_step_percent) * self.num_train_timesteps)
        max_step = int((t_range[1] if t_range else self.max_step_percent) * self.num_train_timesteps)
        
        t = torch.randint(min_step, max_step, (batch_size,), device=input_device)  # 명시적으로 장치 지정
        
        # Sample noise
        noise = torch.randn_like(image_latents, device=input_device)  # 명시적으로 장치 지정
        
        # Add noise (forward diffusion)
        alphas = self.sd_pipeline.scheduler.alphas_cumprod.to(input_device).to(image_latents.dtype)
        sqrt_alphas_t = torch.sqrt(alphas[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_t = torch.sqrt(1 - alphas[t]).view(-1, 1, 1, 1)
        
        noisy_latents = sqrt_alphas_t * image_latents + sqrt_one_minus_alphas_t * noise
        
        print(f"[DEVICE DEBUG] noisy_latents device: {noisy_latents.device}")
        
        # Predict noise with U-Net
        noise_preds = []
        for i in range(batch_size):
            with torch.no_grad():
                latent_input = torch.cat([noisy_latents[i:i+1]] * 2)
                print(f"[DEVICE DEBUG] latent_input device: {latent_input.device}")
                print(f"[DEVICE DEBUG] text_embeddings device: {text_embeddings_list[i].device}")
                
                noise_pred = self.sd_pipeline.unet(
                    latent_input,
                    t[i:i+1],
                    encoder_hidden_states=text_embeddings_list[i]
                ).sample
                
                # Apply guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                g_scale = guidance_scale or self.guidance_scale
                noise_pred_guided = noise_pred_uncond + g_scale * (noise_pred_text - noise_pred_uncond)
                noise_preds.append(noise_pred_guided)
        
        noise_pred_batch = torch.cat(noise_preds, dim=0)
        
        print(f"[DEVICE DEBUG] noise_pred_batch device: {noise_pred_batch.device}")
        
        # Calculate SDS loss
        w = (1 - alphas[t]).view(-1, 1, 1, 1)
        grad_term = w * (noise_pred_batch.detach() - noise)
        
        print(f"[DEVICE DEBUG] grad_term device: {grad_term.device}")
        
        # SDS loss calculation
        sds_loss = torch.sum(grad_term * image_latents) / batch_size
        
        print(f"[DEVICE DEBUG] Final sds_loss device: {sds_loss.device}")
        print(f"[SDS DEBUG] sds_loss.requires_grad: {sds_loss.requires_grad}")
        
        if not sds_loss.requires_grad:
            print(f"ERROR: Final SDS loss does not require grad!")
            return torch.tensor(0.0, requires_grad=True, device=input_device)
        
        return sds_loss.float()