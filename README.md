# Face Detection Using dlib 

This repository contains the code implementing a convenient high-level API for face detection using dlib. 
Admissible files are: images, videos and stream from computer's webcam. In addition to simply detecting the face, it is possible to:
- Blur the detected faces;
- Blur the background; 
- Recognise people whose faces are detected (if examples are provided, of course); 
- Show detection results as the script runs;
- Save detections to disk. 

## Installing dlib

dlib is not particulatly friendly to install, in particular if one wants to enable GPU acceleration. Since dlib uses both CUDA and cuDNN, you need to have *both* installed. After that, create a venv and then proceed as follows:

```
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build
$ cd build
$ cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 
$ cmake --build .
$ cd ..
$ python setup.py install --set DLIB_USE_CUDA=1

```

After that you should have installed dlib with CUDA. To check that this is the case, you could run

```
import dlib
print(dlib.cuda.get_num_devices())
print(dlib.DLIB_USE_CUDA)
```

The first print statement indicates if CUDA-capable devices are actually recognised. However, even if recognised, a CUDA device might not be activate for use with dlib. The second print statement precisely checks that this is the case. Obviously, it is *not* possible to forcefully set `dlib.DLIB_USE_CUDA=True`. 

## Launching the script 

To simply detect faces in a particular file, run

```
python main.py --source <file>
```

If, instead of a file, you pass a directory, e.g. `python main.py --source <directory>` then **all** files in that directory (and subdirectories, too) are recursively processed. Conversely, if you would like to use your webcam, then 

```
python main.py --source webcam
```

In no source flag is passed, the script will automatically read from the `Input` directory I have provided. 

## Options

- `--model <path_to_model>`: the model to be loaded for inference. By default it looks in the `Model` directory, but you can change that at your will. 
- `--upsample <integer>`: if the face is too small, it might not be detected at a first try. To remedy that, you could pass an integer greater than 1: this will upscale the image making it easier to detect small faces. 
- `--show`: when this flag is passed, results of detection will be showed at inference time. Notice: this is the default when webcam is active!
- `--save`: when this flag is passed, results of the dection will be store in the `Output` directory. The name is the same as that of the original file. 
- `--blur <integer>`: if `<integer>` is greater than zero, the face is blurred. The higher the integer, the more violenet the blur is. If `<integer>` is smaller than zero, the background is blurred. 
- `--recognize <path_to_examples_directory>`: it is possible to recognize faces in the picture, provided few pictures of the relevant people are passed. The example directoryis to be organised as follows: 
```
Examples_Directory
│
└── <name_1>
│      │  
│      │ <picture_1>
│      │ <picture_2>
│      │ ...
└── <name_2>
│      │  
│      │ <picture_3>
│      │ <picture_4>
│      │ ...
...
                  
```

## Issues

Due to dlib implementation, you will likely incur in malloc issues. For some reason, the GPU memory will fill up extremely quickly and you will run out of memory. A simple solution could be to resize images. For instance, you could use Image Magick and run

```
mogrify -resize 256x256 ./*.png
```

to resize to 256x256 all .png images in subdirectories of the directory you are launching the script from.v