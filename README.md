# EuresysCamCapture
A python app for camera capturing from a Euresys grabber (Coaxlink Quad G3): live view with frames recording based on OpenCV.

Author: Artem Lutov &lt;&#108;u&#97;&commat;&#108;ut&#97;n.ch&gt;
License: [Apache License, Version 2](www.apache.org/licenses/LICENSE-2.0.html)

## Requirements

1. Install [Euresys Grabber drivers for Coaxlink](https://www.euresys.com/en/Support/Download-area?Series=105d06c5-6ad9-42ff-b7ce-622585ce607f) >= v13
2. Prepare Environment on Linux:
```sh
. /opt/euresys/egrabber/shell/setup_gentl_paths.sh
export GENICAM_GENTL64_PATH=/opt/euresys/egrabber/lib/x86_64:$GENICAM_GENTL64_PATH
export EURESYS_COAXLINK_GENTL64_CTI=/opt/euresys/egrabber/lib/x86_64/coaxlink.cti
export EURESYS_EGRABBER_LIB64=/opt/euresys/egrabber/lib/x86_64
export EURESYS_DEFAULT_GENTL_PRODUCER=coaxlink
```
3. Install Python bindings for eGrabber:
```sh
$ python -m pip install /opt/euresys/egrabber/python/egrabber-13.0.1.32-py2.py3-none-any.whl
```

## Usage
```sh
$ ./camcapture.py -h
usage: camcapture.py [-h] [-f FPS] [-i {png,tiff}] [-n NFRAMES] [-o OUTP_DIR]

Document Taxonomy Builder.

optional arguments:
  -h, --help            show this help message and exit
  -f FPS, --fps FPS     Capturing speed (Frames Per Second) (default: 5)
  -i {png,tiff}, --img-format {png,tiff}
                        Format of the capturing frames (default: png)
  -n NFRAMES, --no-gui-frames-number NFRAMES
                        The number of frames to be captured, being started
                        without any Graphical User Interface (default: 0)
  -o OUTP_DIR, --outp-dir OUTP_DIR
                        Output directory for the captured images (default:
                        imgs)
```

## Examples
Run in the GUI mode (live view), using default settings:
```sh
./camcapture.py
```
Run in the console mode, record 100 frames in TIFF format (near real-time encoding, but consumes 3x more space than PNG) on 10 FPS to the `otiff/` directory:
```sh
./camcapture.py -n 100 -i tiff -f 10 -o otiff
```
