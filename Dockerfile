
RUN pip install realesrgan stable_diffusion_videos[realesrgan]
RUN git config --global credential.helper store
RUN mkdir -p /usr/local/bin
COPY *.py /usr/local/bin/
ENTRYPOINT ["python"]
