#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: Camera capturing app for a Eursys Coaxlink camera grabber
:Authors: (c) Artem Lutov <lua@lutan.ch>, Valentyna Pryhodiuk <vpryhodiuk@lumais.com>
:Date: 2020-10-21
"""
import os
import sys
import time
import cv2
import asyncio
import numpy as np
from functools import partial
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from enum import Enum, auto
from egrabber import *

# Drawing rect to define the roiRect
drawingRect = None

# ROI coordinates (x0, y0), (x1, y1) of the top left and bottom right corners
roiRect = None

# Video  output variables
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
initVideo = None
vidout = None
wTitle = 'Press ESC to exit, SPACEBAR to record'  # Main window
tbExposure = "Exposure time"  # Exposure trackbar in the main window

class ImageFormat(Enum):
	"""Image formats to store the capturing frames"""
	PNG = auto()
	TIFF = auto()
	BMP = auto()
	MP4 = auto()

	def __str__(self):
		return self.name.lower()


def rgb8_to_ndarray(rgb, w, h):
	data = cast(rgb.get_address(), POINTER(c_ubyte * rgb.get_buffer_size())).contents
	c = 3  # 3 channels
	return np.frombuffer(data, count=rgb.get_buffer_size(), dtype=np.uint8).reshape((h,w,c))


async def recordFrame(rgb, outdir, framenum, imgfmt):
	"""Save a frame to the disk asynchronously
	
	rgb: egrabber.ConvertedBuffer | numpy.ndarray  - egrabber buffer in the required RGB format or OpenCV image
	outdir: str  - output directory
	framenum: uint  - frame number
	imgfmt: str  - image format of the output frames
	"""
	vid = imgfmt == str(ImageFormat.MP4)  # Video output
	if isinstance(rgb, egrabber.ConvertedBuffer):
		assert not (roiRect or vid), 'rgb buffer shout not be used with ROI or in video mode'
		rgb.save_to_disk(os.path.join(outdir, 'frame-{:03}.{}'.format(framenum, imgfmt)))  # tiff
	else:
		if roiRect:
			rgb = rgb[roiRect[0][1]:roiRect[1][1], roiRect[0][0]:roiRect[1][0]]
		if vid:
			# print(' [saving video frame: ({}, {}) / {}]'.format(rgb.shape[1], rgb.shape[0], roiRect))
			vidout.write(rgb)
		else:
			cv2.imwrite(os.path.join(outdir, 'frame-{:03}.{}'.format(framenum, imgfmt)), rgb)


def setRoi(event, x, y, flags, params):
	"""Mouse callback to set ROI
	
	event  - mouse event
	x: int  - x coordinate
	x: int  - y coordinate
	flags  - additional flags
	params - extra parameters
	"""
	global drawingRect, roiRect, initVideo
	if event == cv2.EVENT_LBUTTONDOWN:
		drawingRect = [(x, y), (x, y)]
		# print('\nStrting drawingRect: ', drawingRect)
	elif event == cv2.EVENT_LBUTTONUP:
		pTL = (min(drawingRect[0][0], drawingRect[1][0]), min(drawingRect[0][1], drawingRect[1][1]))
		pBR = (max(drawingRect[0][0], drawingRect[1][0]), max(drawingRect[0][1], drawingRect[1][1]))
		drawingRect = None
		# Set the roiRect or reset it
		if pTL != pBR or roiRect:
			initVideo = True  # Init video output
		if pTL != pBR:
			# Adjust to X px padding
			pxBlock = 8
			dx = (pBR[0] - pTL[0]) % pxBlock
			if dx:
				dx = pxBlock - dx
			dy = (pBR[1] - pTL[1]) % pxBlock
			if dy:
				dy = pxBlock - dy
			roiRect = (pTL, (pBR[0] + dx, pBR[1] + dy))
			# print('\nROI size is corrected by ({}, {}): ({}, {})'.format(dx, dy, roiRect[1][0]-roiRect[0][0], roiRect[1][1]-roiRect[0][1]))
		else:
			roiRect = None			

	if event == cv2.EVENT_RBUTTONUP:
		drawingRect = None
		roiRect = None

	if event == cv2.EVENT_MOUSEMOVE and drawingRect:
		drawingRect[1] = (x, y)


def setExposure(grabber, val):
	grabber.remote.set('ExposureTime', val * 1e3)

# , vidout
async def loop(grabber, nframes, tpf, outdir, imgfmt, winw):
	"""Capturing loop
	
    grabber  - camera grabber object
    nframes: int  - the number of frames to be captured in the non-GUI mode;
    tpf: float  - min time per frame in seconds
    outdir: str  - output directory for the resulting frames
    imgfmt: str  - image format of the output frames
    winw: int, >= 1  - Initial wide of the GUI window in pixels
    """
	global initVideo, vidout
	frameCount = 0
	record = nframes > 0  # Whether to record the capturing frame as images
	grabber.start()

	if not nframes:
		w0 = winw
		cv2.namedWindow(wTitle, cv2.WINDOW_NORMAL)
		rfont = 0
		cv2.setMouseCallback(wTitle, setRoi)
		# Set exposition control
		cv2.createTrackbar(tbExposure, wTitle, round(1e3 * 0.005), round(1e3 * tpf), partial(setExposure, grabber))
		cv2.setTrackbarPos(tbExposure, wTitle, round(grabber.remote.get('ExposureTime') / 1e3))
	else:
		print('Recorded frames:', end='')
	videoOutp = imgfmt == str(ImageFormat.MP4)
	initVideo = videoOutp  # Init video output
	while True:
		start = time.perf_counter()
		# timeout in milliseconds
		with Buffer(grabber, timeout=1000) as buffer:
			rgb = buffer.convert('BGR8')  # OpenCV uses BGR colors
			if not nframes:
				# Note: initVideo is defined in mouse callback, where imgfmt is unknown
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

				img = rgb8_to_ndarray(rgb, w, h)  # Note: it fetches the memory rather than making a copy
				if drawingRect or record:
					img = img.copy()  # Copy image to not draw in over the original frame
				if roiRect or drawingRect:
					rc = drawingRect if drawingRect else roiRect
					rt = max(1, round(rfont))
					img = cv2.rectangle(img, rc[0], rc[1], (0, 255, 0), rt)
				if record:
					if initVideo and videoOutp:
						if roiRect:
							vw = roiRect[1][0] - roiRect[0][0]
							vh = roiRect[1][1] - roiRect[0][1]
						else:
							vw = w
							vh = h
						print(' [initializing video as: ({}, {})]'.format(vw, vh))
						vidout = cv2.VideoWriter(os.path.join(outdir, 'video-{:03}.{}'.format(frameCount+1, imgfmt)), fourcc, 1./tpf, (vw,vh))
						initVideo = False
					_, _, w0, h0 = cv2.getWindowImageRect(wTitle)
					# img = cv2.resize(img, (w0, h0))
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
			frameCount += 1
			# Ensure that fps does not exceed teh required value
			dt = time.perf_counter() - start
			if dt < tpf:
				time.sleep(tpf - dt)
			# Note: TIFА images take 3.5x more space than PNG, but required much less CPU
			if record:
				vid = imgfmt == str(ImageFormat.MP4)  # Video output
				img = None
				if roiRect or imgfmt == str(ImageFormat.MP4):
					img = rgb8_to_ndarray(rgb, w, h)
				await recordFrame(img if img is not None else rgb, outdir, frameCount, imgfmt)
				print(' ', frameCount, end='', sep='', flush=True)
	print()  # Ensure newline after the frames output


async def wloop(vid, nframes, tpf, outdir, imgfmt, winw):
	"""Capturing loop for webcamera
	
    vid  - OpenCV video device
    nframes: int  - the number of frames to be captured in the non-GUI mode;
    tpf: float  - min time per frame in seconds
    outdir: str  - output directory for the resulting frames
    imgfmt: str  - image format of the output frames
    winw: int, >= 1  - Initial wide of the GUI window in pixels
    """
	global initVideo, vidout
	frameCount = 0
	record = nframes > 0  # Whether to record the capturing frame as images

	if not nframes:
		w0 = winw
		cv2.namedWindow(wTitle, cv2.WINDOW_NORMAL)
		rfont = 0
		cv2.setMouseCallback(wTitle, setRoi)
	else:
		print('Recorded frames:', end='')
	videoOutp = imgfmt == str(ImageFormat.MP4)
	initVideo = videoOutp  # Init video output
	while True:
		start = time.perf_counter()
		# timeout in milliseconds
		ret, frame = vid.read()
		img = frame.copy()
		if not nframes:
			if not rfont:
				h, w = frame.shape[:2]
				rfont = w / w0
				cv2.resizeWindow(wTitle, w0, int(h / rfont))
			#img = cv2.resize(frame, (w0, int(h / rfont)))
			# cv2.imshow(wTitle, img)

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
				elif vidout is not None:
					vidout.release()
			# img = frame.copy()
			if roiRect or drawingRect:
				rc = drawingRect if drawingRect else roiRect
				rt = max(1, round(rfont))
				img = cv2.rectangle(img, rc[0], rc[1], (0, 255, 0), rt)
			if record:
				if initVideo and videoOutp:
					if roiRect:
						vw = roiRect[1][0] - roiRect[0][0]
						vh = roiRect[1][1] - roiRect[0][1]
					else:
						vw = w
						vh = h
					print(' [initializing video as: ({}, {})]'.format(vw, vh))
					vidout = cv2.VideoWriter(os.path.join(outdir, 'video-{:03}.{}'.format(frameCount+1, imgfmt)), fourcc, 1./tpf, (vw,vh))
					initVideo = False
				_, _, w0, h0 = cv2.getWindowImageRect(wTitle)
				# img = cv2.resize(img, (w0, h0))
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
		frameCount += 1
		# Ensure that fps does not exceed teh required value
		dt = time.perf_counter() - start
		# print('dt: {}, tpf: {}'.format(dt, tpf))
		if dt < tpf:
			time.sleep(tpf - dt)
		# Note: TIF images take 3.5x more space than PNG, but required much less CPU
		if record:
			await recordFrame(frame, outdir, frameCount, imgfmt)
			print(' ', frameCount, end='', sep='', flush=True)
	if vidout is not None:
		vidout.release()
	print()  # Ensure newline after the frames output


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
	parser.add_argument('-o', '--outp-dir', default='res',
		help='Output directory for the captured images')
	parser.add_argument('-c', '--camera', type=int, default=0,
		help='The camera index to start the capturing')
	parser.add_argument('-w', '--wnd-width', type=int, default=1600,
		help='Initial wide of the GUI window in pixels')
	parser.add_argument('-r', '--resolution', type=str, default=None,
		help='Show ("x") or set ("<W>x<H>") camera resolution in the format: <WIDTH>x<HEIGHT>, e.g., 800x600')
	parser.add_argument('-e', '--exposure', type=str, default=None,
		help='Show ("x") or set camera exposure mode and optional exposure time in ms, e.g. for FPS >= 25: -e "Timed 40"')
	args = parser.parse_args()
	
	camres = None   
	if args.resolution:
		assert 'x' in args.resolution, 'Invalid format of the argument'
		if args.resolution != 'x':
			camres = tuple(int(x) for x in args.resolution.split('x'))
		else:
			camres = tuple()

	if not os.path.isdir(args.outp_dir):
		os.makedirs(args.outp_dir)

	try:
		gentl = EGenTL()
		grabber = EGrabber(gentl, 0, args.camera)  # EGrabber(gentl, card_ix, device_ix)
		# Set exposition time by FPS:
		# grabber.remote.set('ExposureTime', 1e6 / (args.fps * 2))

		if camres is not None or args.exposure is not None:
			if camres:
				print('Setting camera resolution: ', camres)
				grabber.remote.set('Width', camres[0])
				grabber.remote.set('Height', camres[1])
			elif camres is not None:
				print('Camera resolution: {}x{}'.format(grabber.remote.get('Width'), grabber.remote.get('Height')))
			if args.exposure is not None:
				if args.exposure != 'x':
					camexp = args.exposure.split(None, 1)
					grabber.remote.set("ExposureMode", camexp[0])
					if len(camexp) >= 2: # and camexp[0] == 'Timed':
						exptime = float(camexp[1])
						assert exptime <= 1000. / args.fps, 'Exposition time is too large for the specified FPS'
						grabber.remote.set('ExposureTime', exptime * 1e3)  # Note: internal exposition time is micro sec
				else:
					print('Camera exposure mode: {} {}'.format(grabber.remote.get('ExposureMode'), grabber.remote.get('ExposureTime')/1e3))
			exit(0)

		# # Set default configuration
		# # Camera configuration
		# grabber.remote.set('TriggerMode', 'Off');  # Off;  # Might not be supported in some cameras
		# grabber.remote.set('TriggerSource', 'Line' + str(args.camera));  # Line0, CoaXPress_Trigger_Input
		# grabber.remote.set('ExposureMode', 'Free_Run_Programmable');  # Free_Run_Programmable, Timed
		# # grabber.remote.set('ExposureTime', 1e6 / args.fps)
		# # Frame grabber configuration
		# grabber.device.set('CameraControlMethod', 'NC');  # 'NC'
		# # grabber.device.set('CycleTriggerSource', 'Immediate');
		# # grabber.device.set('CycleTargetPeriod', 1e6 / args.fps);

		# WARNING: such configuration causes exception on capturing (at least, when no strobing signal is raised)
		# # Camera configuration
		# grabber.remote.set('TriggerMode', 'On');  # Off;  # Might not be supported in some cameras
		# grabber.remote.set('TriggerSource', 'CXPin');  # Line0, CoaXPress_Trigger_Input
		# grabber.remote.set('ExposureMode', 'TriggerWidth');  # Free_Run_Programmable, Timed
		# # grabber.remote.set('ExposureTime', 1e6 / args.fps)
		# # Frame grabber configuration
		# grabber.device.set('CameraControlMethod', 'RG');  # 'NC'
		# # grabber.device.set('CycleTriggerSource', 'Immediate');
		# # grabber.device.set('CycleTargetPeriod', 1e6 / args.fps);
		# # StrobeDuration: 1000,
		# # StrobeDelay: 100,

		grabber.realloc_buffers(3)  # 3, 8
		asyncio.run(loop(grabber, args.nframes, 1. / args.fps, args.outp_dir, args.img_format, args.wnd_width))
	except generated.cEGrabber.GenTLException as err:
		# Fallback to a standard web camera
		vid = cv2.VideoCapture(0)

		if camres is not None or args.exposure is not None:
			if camres:
				print('Setting camera resolution: ', camres)
				vid.set(cv2.CAP_PROP_FRAME_WIDTH, camres[0])
				vid.set(cv2.CAP_PROP_FRAME_HEIGHT, camres[1])
			elif camres is not None:
				print('Camera resolution: {}x{}'.format(
					int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
			if args.exposure is not None:
				if args.exposure != 'x':
					camexp = args.exposure.split(None, 1)
					grabber.remote.set("ExposureMode", camexp[0])
					if len(camexp) >= 2: # and camexp[0] == 'Timed':
						exptime = float(camexp[1])
						assert exptime <= 1. / args.fps, 'Exposition time is too large for the specified FPS'
						vid.set(cv2.CAP_PROP_EXPOSURE, 40)
				else:
					print('Camera exposure: {}'.format(vid.get(cv2.CAP_PROP_EXPOSURE)))
			exit(0)

		vid.set(cv2.CAP_PROP_FPS, args.fps)
		asyncio.run(wloop(vid, args.nframes, 1. / args.fps, args.outp_dir, args.img_format, args.wnd_width))
		vid.release()

	if not args.nframes:
		cv2.destroyAllWindows()
