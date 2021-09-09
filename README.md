# EuresysCamCapture
A python app for camera capturing from a Euresys grabber (Coaxlink Quad G3): live view with frames recording based on OpenCV.

Author: Artem Lutov &lt;&#108;u&#97;&commat;&#108;ut&#97;n.ch&gt;, Valentyna Pryhodiuk &lt;v&#112;ryhodiuk&commat;lumais&#46;&#99;om&gt;  
License: [Apache License, Version 2](www.apache.org/licenses/LICENSE-2.0.html)  
Organizations: [UNIFR](https://www.unifr.ch), [Lutov Analytics](https://lutan.ch/), [LUMAIS](http://lumais.com)

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
$ python3 -m pip install /opt/euresys/egrabber/python/egrabber-13.0.1.32-py2.py3-none-any.whl
```

## Usage
```sh
./camcapture.py -h
usage: camcapture.py [-h] [-f FPS] [-i {png,tiff,bmp,mp4}] [-n NFRAMES]
                     [-o OUTP_DIR] [-c CAMERA] [-w WND_WIDTH] [-r RESOLUTION]

Document Taxonomy Builder.

optional arguments:
  -h, --help            show this help message and exit
  -f FPS, --fps FPS     Capturing speed (Frames Per Second) (default: 5)
  -i {png,tiff,bmp,mp4}, --img-format {png,tiff,bmp,mp4}
                        Format of the capturing frames (default: png)
  -n NFRAMES, --no-gui-frames-number NFRAMES
                        The number of frames to be captured, being started
                        without any Graphical User Interface (default: 0)
  -o OUTP_DIR, --outp-dir OUTP_DIR
                        Output directory for the captured images (default:
                        res)
  -c CAMERA, --camera CAMERA
                        The camera index to start the capturing (default: 0)
  -w WND_WIDTH, --wnd-width WND_WIDTH
                        Initial wide of the GUI window in pixels (default:
                        1600)
  -r RESOLUTION, --resolution RESOLUTION
                        Show ("x") or set ("<W>x<H>") camera resolution in the
                        format: <WIDTH>x<HEIGHT>, e.g., 800x600 (default:
                        None)
  -e EXPOSURE, --exposure EXPOSURE
                        Show ("x") or set camera exposure mode and
                        optional exposure time in ms, e.g. for FPS >= 25:
                        -e "Timed 40" (default: None)
```

### GUI Mode

To start/stop the **recording** (video stream of image files saving), press `r` or `spacebar` keys.

It is possible to select the capturing **ROI** using the left mouse button, which is possible even during the recording process, automatically creating the respective output stream. The ROI is selected by pressing the left mouse button, moving the mouse cursor to form the rectangular ROI area, and releasing the left mouse button to complete the ROI selection. A ROI is reset by the left mouse click on a single pixel (without the mouse move).

## Examples
Run in the GUI mode (live view), using default settings:
```sh
./camcapture.py
```

Run in the GUI mode (live view), capturing from camera #`0` in the resolution `4096x3072` and `4` frames per second rate, saving the captured frames as a video stream in the `mp4` format in the directory `./video` (the video is suffixed with the prefix that corresponds to the starting frame):
```sh
./camcapture.py -c 0 -r 4096x3072 -f 4 -i mp4 -o video
```

Run in the console mode, record 100 frames in TIFF format (near real-time encoding, but consumes 3x more space than PNG) on 10 FPS to the `otiff/` directory:
```sh
./camcapture.py -n 100 -i tiff -f 10 -o otiff
```
