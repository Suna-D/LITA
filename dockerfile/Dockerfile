FROM nvcr.io/nvidia/pytorch:22.04-py3
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "workspace:$PYTHONPATH"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev

WORKDIR /workspace
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]