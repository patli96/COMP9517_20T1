# COMP9517 20T0 Project - group: 2020_vision
Project page on course website:
https://webcms3.cse.unsw.edu.au/COMP9517/20T1/resources/41481

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

### Preparation
* Download the YOLOv3 weight here: https://pjreddie.com/media/files/yolov3.weights
and place the file under /pedestrian_monitor/detectors/
* To run the CSP detector, you need to setup the CSP environment.
   * Git clone https://github.com/dominikandreas/CSP
   * Create a separate Python Virtual Environment
   * Install requirements.txt inside, for instructions of installing Tensorflow-gpu, see below.
   * Make sure you have installed CUDA 10.0 and related CuDNN.
   * Install Cython, matplotlib, bottle using pip
   * Put `CSP_utils/pred_server.py` and `CSP_utils/setup_win.py` inside
   * Windows guide:
       * install Visual Studio 2017, not 2019. If you have Visual Studio 2019, you need to uninstall it.
       * in `setup_win.py`, change the VC path inside the file to your path
       * in `setup_win.py`, change Line 52 gpu_nms.cpp to gpu_nms.pyx
       * in `keras_csp/nms/cpu_nms.pyx` and `gpu_nms.pyx`, line 25, change `np.int_t` to `np.intp_t`
       * run: python setup_win.py install
       * in generated `gpu_nms.cpp` , change Line 2166 `__pyx_t_5numpy_int32_t` to `int`
       * in `setup_win.py` , change `gpu_nms.pyx` to `gpu_nms.cpp` to avoid re-generation
       * run: `python setup_win.py install`, it should compile successfully
       * run: `python setup_win.py build_ext --inplace`
       * then you can finally run `python pred_server.py`.

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
   
   To specify preprocessor, detector, tracker or/and clusterer:
   
   ```bash
   python pedestrian_monitor --preprocessor background_subtraction --detector csp_detector --tracker mot_tracker --clusterer trajectory_agglomerative_track
   ```
   
   To display the help information, you may run:
   
   ```bash
   python pedestrian_monitor -h
   ```
   
   Please note that some arguments (i.e. fps) are not implemented yet.

### Evaluations

#### Detectors

Reference: https://github.com/Cartucho/mAP

##### Generating bounding box txt files
run `write_bb_to_file.py` to output bounding boxes as txt files to detection_results folder
modify line:
```
generate_bounding_boxes_file(
   'yolo', # folder name
   0.3, # confidence level
   imgs_modes # image background, if applying image preprocessing
)
```
##### run evaluation against ground_truth
* Place the detection txt files in `/mAP/input/detection-results`
* run `python main.py`
* result will be produced in `/mAP/output` folder

#### Trackers

Reference: https://github.com/cheind/py-motmetrics
