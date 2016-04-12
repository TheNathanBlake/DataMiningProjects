import winsound

def drake():
	freq = [659, 659, 659, 698, 659, 587, 523, 659, 523]
	dur  = [240, 240, 240, 240, 240, 240, 240, 500, 960]

	for i in range(0,len(freq)):
		winsound.Beep(freq[i], dur[i])
