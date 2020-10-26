#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: Camera capturing app for a Eursys Coaxlink camera grabber

:Authors: (c) Artem Lutov <lua@lutan.ch>
:Date: 2020-10-21
"""
import os
import sys
import time
import cv2
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from enum import Enum, auto
from egrabber import *


class ImageFormat(Enum):
	"""Image formats to store the capturing frames"""
	PNG = auto()
	TIFF = auto()

	def __str__(self):
		return self.name.lower()

def rgb8_to_ndarray(rgb, w, h):
	data = cast(rgb.get_address(), POINTER(c_ubyte * rgb.get_buffer_size())).contents
	c = 3  # 3 channels
	return np.frombuffer(data, count=rgb.get_buffer_size(), dtype=np.uint8).reshape((h,w,c))

# , vidout
def loop(grabber, nframes, tpf, outdir, imgfmt):
	"""Capturing loop

    grabber  - camera grabber object
    nframes: int  - the number of frames to be captured in the non-GUI mode;
    tpf: float  - min time per frame in milli seconds
    outdir: str  - output directory for the resulting frames
    imgfmt: str  - image format of the output frames
    """
	frameCount = 0
	record = nframes > 0  # Whether to record the capturing frame as images
	grabber.start()
	if not nframes:
		wTitle = 'Press ESC to exit, SPACEBAR to record'
		w0 = 1600
		cv2.namedWindow(wTitle, cv2.WINDOW_NORMAL)
		rfont = 0
	else:
		print('Recorded frames:', end='')
	while True:
		start = time.perf_counter()
		# timeout in milliseconds
		with Buffer(grabber, timeout=5000) as buffer:
			rgb = buffer.convert('RGB8')
			if not nframes:
				if not rfont:
					w = buffer.get_info(BUFFER_INFO_WIDTH, INFO_DATATYPE_SIZET)
					h = buffer.get_info(BUFFER_INFO_HEIGHT, INFO_DATATYPE_SIZET)
					rfont = w / w0  # Font ratio
					cv2.resizeWindow(wTitle, w0, int(h / rfont))
				key = cv2.waitKey(1) & 0xFF
				# Quit: escape, e or q
				if key in (27, ord('e'), ord('q')):
					break
				## Pause: spacebar or p
				#elif key in (32, ord('p')):
				#	cv2.waitKey(-1)
				# Record: enter, spacebar or r
				elif key in (10, 32, ord('r')):
					record = not record
					print(('' if record else '\n') + 'Recording: ',
						os.path.normpath(outdir) + '/' if record else 'OFF')
					if record:
						print('Recorded frames:', end='')

				img = rgb8_to_ndarray(rgb, w, h)
				if record:
					_, _, w0, _ = cv2.getWindowImageRect(wTitle)
					rfont = max(1, w / w0)
					cv2.putText(img,'R',
						(20, 20 + int(24 * rfont)),  # bottomLeftCornerOfText
						cv2.FONT_HERSHEY_SIMPLEX,
						rfont,  # Font size
						(7, 7, 255),  # Color
						1 + round(rfont)) # Line thickness, px
				cv2.imshow(wTitle, img)
			elif frameCount == nframes:
				break
			# Note: TIF images take 3.5x more space than PNG, but required much less CPU
			if record:
				rgb.save_to_disk(os.path.join(outdir, 'frame.{:03}.{}'.format(frameCount, imgfmt)))  # tiff
				print(' ', frameCount, end='', sep='', flush=True)
			# img = rgb8_to_ndarray(rgb, w, h)
			# vidout.write(img)
			frameCount += 1
			# Ensure that fps does not exceed teh required value
			dt = time.perf_counter() - start
			if dt < tpf:
				time.sleep(dt / 1000)
	print()  # Ensure newline after teh frames output

def run(grabber, nframes, fps, outdir, imgfmt):
	grabber.realloc_buffers(5)  # 3
	# w = grabber.get_width()
	# h = grabber.get_height()
	# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	# outdir = os.path.splitext(__file__)[0] + '.output'
	if not os.path.isdir(outdir):
		os.makedirs(outdir)
	# out = cv2.VideoWriter(os.path.join(outdir, 'output.avi'), fourcc, fps, (w,  h))
	loop(grabber, nframes, 1000 / fps, outdir, imgfmt)
	if not nframes:
		cv2.destroyAllWindows()

if __name__ == '__main__':
	parser = ArgumentParser(description='Document Taxonomy Builder.',
		formatter_class=ArgumentDefaultsHelpFormatter,
		conflict_handler='resolve')
	parser.add_argument('-f', '--fps', type=int, default=5,
		help='Capturing speed (Frames Per Second)')
	parser.add_argument('-i''', '--img-format', type=str, default=ImageFormat.PNG,
		choices=[str(v) for v in ImageFormat], help='Format of the capturing frames')
	parser.add_argument('-n', '--no-gui-frames-number', dest='nframes', type=int, default=0,
		# help='Caprute frames from the terminal without any Graphical User Interface')
		help='The number of frames to be captured, being started without any Graphical User Interface')
	parser.add_argument('-o', '--outp-dir', default='imgs',
		help='Output directory for the captured images')
	args = parser.parse_args()

	gentl = EGenTL()
	grabber = EGrabber(gentl)
	run(grabber, args.nframes, args.fps, args.outp_dir, args.img_format)
