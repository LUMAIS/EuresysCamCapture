async def wloop(nframes, tpf, outdir, imgfmt):
	"""Capturing loop

    nframes: int  - the number of frames to be captured in the non-GUI mode;
    tpf: float  - min time per frame in milli seconds
    outdir: str  - output directory for the resulting frames
    imgfmt: str  - image format of the output frames
    """
	wTitle = 'Press ESC to exit, SPACEBAR to record'
	frameCount = 0
	record = nframes > 0  # Whether to record the capturing frame as images

	if not nframes:
		wTitle = 'Press ESC to exit, SPACEBAR to record'
		w0 = 1600
		cv2.namedWindow(wTitle, cv2.WINDOW_NORMAL)
		rfont = 0
		#cv2.setMouseCallback(wTitle, setRoi)
	else:
		print('Recorded frames:', end='')
	while True:
		start = time.perf_counter()
		# timeout in milliseconds
		vid = cv2.VideoCapture(0)
		ret, frame = vid.read()
		img = frame.copy()
		if not nframes:
			if not rfont:
				h, w = frame.shape[:2]
				rfont = w / w0
				cv2.resizeWindow(wTitle, w0, int(h / rfont))
			#img = cv2.resize(frame, (w0, int(h / rfont)))
			cv2.imshow(wTitle, img)

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
			img = frame.copy()
			if record:
				_, _, w0, h0 = cv2.getWindowImageRect(wTitle)
				img = cv2.resize(img, (w0, h0))
				rfont = 1  # max(1, w / w0)
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
			time.sleep(dt / 1000)
		# Note: TIF images take 3.5x more space than PNG, but required much less CPU
		if record:
			await recordFrame(img, outdir, frameCount, imgfmt)
			print(' ', frameCount, end='', sep='', flush=True)

	print()  # Ensure newline after the frames output