FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install jupyter \
                ipdb \
		        matplotlib \
		        sklearn \
		        pandas \
RUN pip install git+https://github.com/lanpa/tensorboardX