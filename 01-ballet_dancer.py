from os import environ

import torch

from huggingface_hub import notebook_login
from huggingface_hub.hf_api import HfFolder
from stable_diffusion_videos import StableDiffusionWalkPipeline, Interface

if __name__ == '__main__':
  HfFolder.save_token(environ.get('HF_TOKEN'))
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
  interface = Interface(pipeline)
  # Seconds in the song

  artists =  ' and '.join([‘ivan aivazovsky’, ‘greg rutkowski’, ‘rutkowski’])
  cues = '. '.join([‘digital art’, ‘hyper detailed, sharp focus, soft light’, ‘octane render’, ‘ray tracing’, ‘trending on artstation’])
  container = ‘beautiful painting’
  prompts = [
    f’A {container} of {target} by {artists}, in style of {cues}’
    for target in [
      'a ballet dancer girl dreaming of a city',
      'a ballet dancer girl walking in a city',
      'a ballet dancer girl watching a big city building',
      'a ballet dancer girl dancing a big city building',
      'a big city building turning into a huge burger',
      'a ballet dancer girl eating a huge burger',
      'a huge burger turning into a man shadow',
      'a ballet dancer girl watching a man shadow leaving',
      'a ballet dancer girl crying in a city',
      'a ballet dancer girl disappearing in dust',
      # 0 to 45
      'a ballet dancer girl spirit into space',
      'a ballet dancer girl dancing with stars',
      'a ballet dancer girl melting',
      '',
      'a big city building turning into a huge burger',
      'a ballet dancer girl eating a huge burger',
      'a huge burger turning into a man shadow',
      'a ballet dancer girl watching a man shadow leaving',
      'a ballet dancer girl crying in a city',
      'a ballet dancer girl disappearing in dust',
    ]
  ]
 
  audio_offsets = list(range(0, 50, 5))
  fps = 30
  num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]
  seeds = [random.randint(10, 6954010) for _ in range(len(prompts))]
 
  video_path = pipeline.walk(
      prompts=prompts,
      seeds=seeds,
      num_interpolation_steps=num_interpolation_steps,
      height=512,                            # use multiples of 64
      width=512,                             # use multiples of 64
      audio_filepath='/workdir/audio.mp3',  # Use your own file
      audio_start_sec=audio_offsets[0],      # Start second of the provided audio
      fps=fps,                               # important to set yourself based on the num_interpolation_steps you defined
      batch_size=2,                          # increase until you go out of memory
      output_dir='/workdir',                 # Where images will be saved
      name=None,                             # Subdir of output dir. will be timestamp by default
      upsample=True,
  )
  print(f'Video generated at {video_path}')
