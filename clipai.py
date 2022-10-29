from pathlib import Path
from typing import List

import torch
import typer
import yaml

from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler
from huggingface_hub.hf_api import HfFolder
from pydantic import BaseModel, BaseSettings
from stable_diffusion_videos import StableDiffusionWalkPipeline
from yaml.loader import SafeLoader


class Settings(BaseSettings):
    HF_TOKEN: str


class RenderingStyle(BaseModel):
    artists: List[str]
    container: str
    cues: List[str]

    def to_prompt_template(self) -> str:
        return ''.join([
            'A ',
            self.container,
            ' of {0} by ',
            ' and '.join(self.artists),
            ', in style of ',
            '. '.join(self.cues)
        ])


class PromptConfiguration(BaseModel):
    duration: int = 10
    sentences: List[str]
    seeds: List[int]


class RenderingConfiguration(BaseModel):
    name: str
    duration: int
    fps: int = 30
    style: RenderingStyle
    prompts: PromptConfiguration


def entrypoint(
    audio_path: Path = typer.Option(...),
    configuration_path: Path = typer.Option(...),
    output_directory: Path = typer.Option(...),
    batch_size: int = typer.Option(2),
    resume: bool = typer.Option(False),
) -> None:
    print(':: authenticate to HF')
    settings = Settings()
    HfFolder.save_token(settings.HF_TOKEN)
    print(':: loading model')
    pipeline = StableDiffusionWalkPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        vae=AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema"),
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
        scheduler=LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
    ).to("cuda")
    print(':: load configuration')
    with configuration_path.open() as stream:
        configuration = RenderingConfiguration(**yaml.load(stream, Loader=SafeLoader))
    offsets = list(range(0, configuration.duration, configuration.prompts.duration))
    steps =  [(b-a) * configuration.fps for a, b in zip(offsets, offsets[1:])]
    template = configuration.style.to_prompt_template()
    prompts = [template.format(sentence) for sentence in configuration.prompts.sentences]
    print(':: start inference')
    path = pipeline.walk(
        prompts=prompts,
        seeds=configuration.prompts.seeds,
        num_inference_steps=50,
        guidance_scale=10,
        margin=1.0,
        smooth=0.2,
        resume=resume,
        upsample=True,
        num_interpolation_steps=steps,
        height=512,
        width=512,
        audio_filepath=str(audio_path),
        audio_start_sec=offsets[0],
        fps=configuration.fps,
        batch_size=batch_size,
        output_dir=str(output_directory),
        name=configuration.name,
    )
    print(f':: video generated at {path}')


if __name__ == '__main__':
    typer.run(entrypoint)