# COMP9517 20T0 Project - group: 2020_vision
Project page on course website:
https://webcms3.cse.unsw.edu.au/COMP9517/20T1/resources/41481

## Overview
TBD

---

University of New South Wales - Term 1, 2020  
COMP9517 - Computer Vision
Assignment 2 - Team 2020_vision

### Team members

| Name                  | zID      | Email                          |
| :-------------------- | :------- | :----------------------------- |
| Alexander Catic       | z3252894 | a.catic@student.unsw.edu.au    |
| Alexander Bunn        | z5146667 | z5146667@ad.unsw.edu.au        |
| Daheng (Harry) Wang   | z5234730 | 0w0@wdhwg001.me                |
| Bingquan (James) Wang | z5228822 | jameswang1019@gmail.com        |
| Patrick Li            | z5180847 | patrick.li@student.unsw.edu.au |

### Communication

Slack: https://2020visionco.slack.com/

### Running code

1. Set up the environment.

   * For Poetry:
   
     This is the recommended way of setting up the environment,
      because it allows you to easily remove unwanted or incompatible packages.
     
     1. Install Poetry: https://python-poetry.org/docs/#installation
     2. Run `poetry install` to install dependencies.
        
        Please note that if you are using AMD or Intel Graphics Card,
         you may need to remove `tensorflow-gpu` from `pyproject.toml`.
        
        And if you are using NVIDIA Graphics Card, please make sure
         you've installed the following packages:
        * The latest driver: https://www.nvidia.com/download/index.aspx?lang=en-us
        * cuDNN SDK: https://developer.nvidia.com/cudnn
        * CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit-archive
        
        For other issues of using Tensorflow GPU, you may check here:
         https://www.tensorflow.org/install/gpu
      3. Run `poetry shell` to enter the visual environment.

   * For Anaconda:

      1. Run `conda env create -f environment.yml` to install a new environment.

         Please refer to the content above if you have any issues related to Tensorflow.

         This `environment.yml` is generated on Windows, so Linux and macOS users may have
          trouble installing some packages. In this case, you may need to create the
           Anaconda environment manually and install all packages mentioned in
            `pyproject.toml`.
      2. Run `conda activate COMP9517_Project` to activate the installed environment.

   * Others:

      Although not recommended, the project still provides the traditional
       `requirements.txt` file to manually set up a visual environment.

2. Run the project.

   ```bash
   python pedestrian_monitor
   ```

   Or you can also run it as a module:

   ```bash
   python -m pedestrian_monitor
   ```
   
   To display the help information, you may run:
   
   ```bash
   python pedestrian_monitor -h
   ```
   
   Please note that some arguments are not implemented yet.
