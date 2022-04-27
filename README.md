This repo contains a solution to NeuroWood hackathon.

Training is based on "tf_efficientnetv2_s_in21ft1k" model from "timm" library. This model was first trained on Imagenet21k 
and then finetuned on Imagenet1k thus providing extra generalization power. Overfitting is combatted by using a diverse set of
augmentation techniques from Albumentation library. Another significant component of training is fmix augmentation that 
combines parts of 2 images with different class labels and calculates loss based on ratio of each image in the combined one. 
To ensure faster training OneCycle schedule is used.

Original data is assumed to be located in "data/" folder with "1", "3", "drova", "test" and "test_top_scores" subfolders.
Semantic contents of images is not very diverse and most features are rather large in size plus most modern image 
classification algorithms are rather memory-hungry so downsampling to (512, 512) images makes sense. Downsampling is 
made in a naive way by resizing image so that the shortest side is 512 pixels in length and then taking central crop 
of 512 pixels along longest side. Tests confirming correct data preparation is provided.

To reproduce the experiments and then prepare a submit to the hackathon in "data/test.csv" file you need to run the command:
```
docker run -it --gpus device=0 --name pytorch --shm-size 16gb -v ${PWD}:/workspace tetelias/conda-pytorch:v1 bash run-training-then-submit.sh
```

To simply prepare a submit to the hackathon in "test.csv" file you need to run the command:
```
docker run -it --gpus device=0 --name pytorch --shm-size 16gb -v ${PWD}:/workspace tetelias/conda-pytorch:v1 bash run-submit.sh
```

After the end of competition an extra test data was provided. To generate predictions on this set you need to uncomment last line in either 
"*.sh" files and "data/test_top_scores.csv" will contain the labels.