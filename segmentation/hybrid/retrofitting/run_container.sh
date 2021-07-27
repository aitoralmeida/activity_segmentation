#!/bin/bash

sudo docker stop ubermejo-retrofitting
sudo docker rm ubermejo-retrofitting
sudo docker run -it --ipc=host --gpus "device=0" --name ubermejo-retrofitting -v activity-segmentation:/results ubermejo/retrofitting bin/bash
