To build an image:

    sudo docker build -t starspace --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

To run a container:

    sudo docker run --rm -e starspace_input_file=starspace_input_file.txt -e model_output_file=model_output_file -v /home/aleksandr/MLEng/containerization_hw/volume:/src/volume starspace
