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

# env variables and volume
ENV STARSPACE_INPUT_FILE starspace_input_file.txt
ENV MODEL_OUTPUT_FILE model_output_file
VOLUME /volume

# create non-root user
RUN groupadd -g 1001 joker
RUN useradd -ms /bin/bash -u 1001 -g 1001 newuser
USER newuser

ENTRYPOINT ["./starspace_script.sh"]