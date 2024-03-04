# bct-ca-motion

This project is a part of the thesis: Deep Learning-Based Analysis and Motion Tracking of Calcium Transient in Cardiac Tissues.
This is an implementation of a Python pipeline dedicated for motion dynamics analysis of calcium transients in bioartificial cardiac tissues.

### Required libraries
___
The following libraries are needed for the pipeline to run.
1. [OpenCV](https://opencv.org/get-started/)
2. [Pytorh](https://pytorch.org/get-started/locally/)
3. [Kornia](https://github.com/kornia/kornia)
4. [alive-progress](https://github.com/rsalmei/alive-progress)

The pipeline works on Windows, Linux, and MacOS. Below are sample installations of the above libraries for Linux Ubuntu 22.04.4 LTS using [Pip](https://pypi.org/project/pip/) and [Anaconda](https://docs.anaconda.com/).
```
conda install opencv

conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly

conda install conda-forge::kornia

pip install alive-progress
```
