import torch

from huggingface_hub import notebook_login
from stable_diffusion_videos import StableDiffusionWalkPipeline, Interface

if __name__ == '__main__':
  pipeline = StableDiffusionWalkPipeline.from_pretrained(
      "CompVis/stable-diffusion-v1-4",
      torch_dtype=torch.float16,
      revision="fp16",
  ).to("cuda")
  interface = Interface(pipeline)
  # Seconds in the song
  duration = 240
  audio_offsets = [0, 5, 10, 15, 20, 25, 30, 35, 40] # list(range(0, duration, 2))
  qualifiers = ','.join(['anime', 'ghibli', 'oil painting' 'high resolution', 'detailed', '4K', 'vivid colors'])
  sequences = [
      'a ballet dancer girl dreaming of a big city streets',
      'a ballet dancer girl walking in a big city streets',
      'a ballet dancer girl watching a building',
      'a ballet dancer girl dancing with a building',
      'a building turning into a huge burger',
      'a ballet dancer girl eating a burger',
      'a burger turning into a man shadow',
      'a ballet dancer girl watching a man shadow leaving',
      'a sad ballet dancer girl crying',
      'a ballet dancer girl disappearing in dust'
  ]
  prompts = [
      f'{sequence},{qualifiers}'
      for sequence in sequences
  ]
  fps = 25
  # Convert seconds to frames
  num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]
  video_path = pipeline.walk(
      prompts=prompts,
      seeds=[42, 1337],
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
