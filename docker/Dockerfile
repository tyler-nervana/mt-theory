FROM pytorch/pytorch:latest

# Install interactive work dependencies
RUN apt-get update && \
    apt-get install -y emacs24 \
                       tmux

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install jupyter \
                ipdb \
		matplotlib \
		sklearn \
		pandas \
		librosa
RUN pip install git+https://github.com/pytorch/ignite
RUN pip install git+https://github.com/lanpa/tensorboardX