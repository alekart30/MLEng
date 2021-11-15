FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git
RUN apt install -y build-essential

# install Boost library
RUN wget -O boost_1_63_0.zip https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.zip/download
RUN unzip boost_1_63_0.zip
RUN mv boost_1_63_0 /usr/local/bin

# install starspace
RUN git clone https://github.com/facebookresearch/Starspace.git
WORKDIR /Starspace
RUN make

# copy source files
WORKDIR /
RUN mkdir /src
COPY starspace_script.sh /src
WORKDIR /src

ENV starspace_input_file starspace_input_file.txt
ENV model_output_file model_output_file
VOLUME /volume

# create non-root user
RUN useradd -ms /bin/bash newuser
USER newuser

ENTRYPOINT ["./starspace_script.sh"]