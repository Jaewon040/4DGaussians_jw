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
            잠재 표현
        """
        # VAE를 GPU로 이동 (필요한 경우)
        self.sd_pipeline.vae = self.sd_pipeline.vae.to(self.device)
        
        # 배치 처리
        latents_list = []
        batch_size = images.shape[0]
        
        # 메모리 제한을 위해 작은 배치로 처리
        sub_batch_size = 4
        for i in range(0, batch_size, sub_batch_size):
            sub_batch = images[i:i+sub_batch_size]
            with torch.no_grad():
                latents = self.sd_pipeline.vae.encode(sub_batch.to(torch.float16)).latent_dist.sample()
                latents = 0.18215 * latents  # SD 모델을 위한 스케일링
                latents_list.append(latents)
                
        # VAE를 다시 CPU로 이동
        self.sd_pipeline.vae = self.sd_pipeline.vae.to("cpu")
        
        return torch.cat(latents_list, dim=0)
    
    
    # Modified StableDiffusionSDS.forward method to handle dtype mismatches
    def forward(
        self,
        rendered_images: torch.Tensor,
        prompt: Union[str, List[str]],
        negative_prompt: str = "",
        guidance_scale: Optional[float] = None,
        t_range: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        batch_size = rendered_images.shape[0]
        
        # Ensure images are in the correct range and dtype
        if rendered_images.min() < 0 or rendered_images.max() > 1:
            print(f"Warning: Input image range outside [0, 1]. Min: {rendered_images.min()}, Max: {rendered_images.max()}")
            rendered_images = torch.clamp(rendered_images, 0.0, 1.0)
            
        # Convert to float16 to match model's dtype requirements
        rendered_images_fp16 = rendered_images.to(torch.float16)
        
        # Scale to [-1, 1] range for SD model
        rendered_images_fp16 = 2 * rendered_images_fp16 - 1
        
        # Process prompts
        prompt_list = [prompt] * batch_size if isinstance(prompt, str) else prompt
        neg_prompt_list = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
        
        total_loss = 0.0
        
        for i in range(batch_size):
            # Process single image
            single_image = rendered_images_fp16[i:i+1]
            single_prompt = prompt_list[i] if i < len(prompt_list) else prompt_list[0]
            single_neg_prompt = neg_prompt_list[i] if i < len(neg_prompt_list) else neg_prompt_list[0]
            
            # Get text embeddings
            text_embeddings = self.get_text_embeddings(single_prompt, single_neg_prompt)
            
            # Encode image to latent space
            with torch.no_grad():
                image_latents = self.encode_images(single_image)
                
            # Sample timestep
            min_step = int((t_range[0] if t_range else self.min_step_percent) * self.num_train_timesteps)
            max_step = int((t_range[1] if t_range else self.max_step_percent) * self.num_train_timesteps)
            
            t = torch.randint(min_step, max_step, (1,), device=self.device)
            
            # Sample noise
            noise = torch.randn_like(image_latents)
            
            # Add noise (forward diffusion)
            alphas = self.sd_pipeline.scheduler.alphas_cumprod.to(self.device).to(image_latents.dtype)
            alphas_t = alphas[t]
            sqrt_alphas_t = torch.sqrt(alphas_t).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_t = torch.sqrt(1 - alphas_t).view(-1, 1, 1, 1)
            
            noisy_latents = sqrt_alphas_t * image_latents + sqrt_one_minus_alphas_t * noise
            
            # Predict noise with U-Net
            with torch.no_grad():
                latent_model_input = torch.cat([noisy_latents] * 2)
                noise_pred = self.sd_pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Apply guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                g_scale = guidance_scale or self.guidance_scale
                noise_pred = noise_pred_uncond + g_scale * (noise_pred_text - noise_pred_uncond)
            
            # Calculate SDS gradient
            w = (1 - alphas_t).view(-1, 1, 1, 1)
            grad = w * (noise_pred - noise)
            
            # SDS loss calculation - make sure everything is same dtype
            zero_target = torch.zeros_like(grad, dtype=grad.dtype, device=grad.device)
            single_loss = torch.nn.functional.mse_loss(grad, zero_target, reduction='mean')
            total_loss += single_loss
        
        # Return the average loss with proper dtype conversion
        # Convert back to float32 for the rest of the training
        return (total_loss / batch_size).to(torch.float32)