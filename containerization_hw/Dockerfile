FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    && apt install -y build-essential

# install Boost library
RUN wget -O boost_1_63_0.zip https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.zip/download \
    && unzip boost_1_63_0.zip \
    && mv boost_1_63_0 /usr/local/bin

# install starspace
RUN git clone https://github.com/facebookresearch/Starspace.git \
    && cd Starspace/ \
    && make

# copy source files
RUN mkdir /src
COPY starspace_script.sh /src
WORKDIR /src
RUN chmod +x starspace_script.sh

# env variables and volume
ENV STARSPACE_INPUT_FILE starspace_input_file.txt
ENV MODEL_OUTPUT_FILE model_output_file
ARG USER_ID
ARG GROUP_ID
VOLUME /volume

# create non-root user
RUN groupadd -g $GROUP_ID group \
    && useradd -ms /bin/bash -u $USER_ID -g $GROUP_ID user
USER user

ENTRYPOINT ["./starspace_script.sh"]