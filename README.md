# bct-ca-motion

This project is a part of the thesis: Deep Learning-Based Analysis and Motion Tracking of Calcium Transient in Cardiac Tissues.
This is an implementation of a Python pipeline dedicated for motion dynamics analysis of calcium transients in bioartificial cardiac tissues.

## Required libraries

Installing the following libraries is necessary for the pipeline to run.
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
In addition to the above installations, it is necessary to clone [Torch-em](https://github.com/constantinpape/torch-em) as well as its dependency [Elf](https://github.com/constantinpape/elf).

## Folder structure

After cloning [bct-ca-motion](https://github.com/maielshazly-git/bct-ca-motion), 4 more folders should be placed in the same directory of the source code:
1. Change the name of the cloned directory of [Torch-em](https://github.com/constantinpape/torch-em) to "torch_em".
2. Change the name of the cloned directory of [Elf](https://github.com/constantinpape/elf) to "elf".
3. Download the mask generation (and optionally the mask enhancement) model checkpoint(s) from the URL provided in [segmentation-dnns/ReadMe.md](segmentation-dnns/ReadMe.md).
4. Place the model(s) in a directory under the name "checkpoints".
5. Place the above 3 directories as well as the directory containing the input image sequence in the same directory of the source code acquired from [src](src).

The directory layout should be similar to the following:

bct-ca-motion <br />
└── checkpoints <br />
&emsp;&emsp;&nbsp;├── mask-enhancer.pt <br />
&emsp;&emsp;&nbsp;├── mask-generator.pt <br />
└── elf <br />
&emsp;&emsp;&nbsp;└── ... <br />
└── torch_em <br />
&emsp;&emsp;&nbsp;└── ... <br />
└── images <br />
&emsp;&emsp;&nbsp;├── ... <br />
├── flows_averaging_handler.py <br />
├── fluorescence_signals_processor.py <br />
├── ... <br />
<br />
After completing the execution of the pipeline, all output directories will be automatically placed in one directory called "Output". The output parent directory will be placed in the same directory of the source code; for example, bct-ca-motion in the layout shown above.

## Commands and paramaeters

Navigate to the code directory and run the following command to see a detailed description of all required and optional parameters.
```
cd bct-ca-motion

python main.py --help
```
The input directory of the image sequence and the number of frames per second are the required parameters. Providing them is sufficient for running the pipeline.
```
python main.py --input_dir 'images' --frames_per_sec 100
```
Optional parameters can be added to replace the default settings of the pipeline. For example, the following command prompts the code to use multiprocessing with 4 CPUs to smooth the normalized images.
```
python main.py --input_dir 'images' --frames_per_sec 100 --smooth_multiprocess --cpu_count 4
```

## Sample tracking of calcium transients

<p align="center">
  <img src="sample-output/e-4031_3nM.gif" width="300" title="BCT with low E-4031">
  <img src="sample-output/ca_0.8mM.gif" width="300" title="BCT with low calcium">
  <img src="sample-output/nif_10.0uM.gif" width="300" title="BCT with high Nifedipine">
</p>
