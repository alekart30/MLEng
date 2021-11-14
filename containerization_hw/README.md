To build an image:

    sudo docker build -t starspace .

To run a container:

    sudo docker run --rm -e starspace_input_file=starspace_input_file.txt -e model_output_file=model_output_file -v /media/sf_MLEngCourse/MLEng/containerization_hw/volume:/src/volume starspace
