## Train
**Attention:** Train step does not work "out of the box" from the github repo, because I did not add data preparation scripts.

## Test
1. Clone the repository
2. Build Docker image
"cd" to repository folder and run next command in the terminal
```
docker build -t <your-image-name> .
```
3. Run test.py inside the Docker image

**Example:**
- `/home/user/images` - path to directory with images on the local PC
- `/home/user/masks` - path to directory where masks should be saved (**path must exist**)

Then run next command to start `test.py` with default threshold (0.5) and model from selected_outputs (UNet_celeba_hq_HairMatting_28062020_101930)
```
docker run -it --rm -v /home/user/images:/data -v /home/user/masks:/output <your-image-name> python test.py --path-to-images /data --path-to-masks /output
```

### Test script parameters
```
usage: test.py [-h]
               [--experiment-name {UNet_celeba_hq_BCEWithLogits_28062020_101854,UNet_celeba_hq_HairMatting_28062020_101930}]
               [--model-type {UNet}] [--path-to-images PATH_TO_IMAGES]
               [--path-to-masks PATH_TO_MASKS] [--threshold THRESHOLD]

Test hair segmentator on dir with images

optional arguments:
  -h, --help            show this help message and exit
  --experiment-name {UNet_celeba_hq_BCEWithLogits_28062020_101854,UNet_celeba_hq_HairMatting_28062020_101930}
                        Experiment name. Used to load model
  --model-type {UNet}   Model type. Used to load model state_dict
  --path-to-images PATH_TO_IMAGES
                        Path to the folder with images to segment
  --path-to-masks PATH_TO_MASKS
                        Path to dir where masks should be saved
  --threshold THRESHOLD
                        Threshold of segmentation
```

## Datasets
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- [Figaro-1k](http://projects.i-ctm.eu/it/progetto/figaro-1k)

## Papers
- [Boundary-Aware Network for Fast and High-Accuracy Portrait Segmentation](https://arxiv.org/abs/1901.03814) (not completed)
- [Real-time deep hair matting on mobile devices](https://arxiv.org/abs/1712.07168)
- [Real-time Hair Segmentation and Recoloring on Mobile GPUs](https://arxiv.org/abs/1907.06740)