import winsound

# You used to debug on Ubuntu
def drake():
	freq = [659, 659, 659, 698, 659, 587, 523, 659, 523]
	dur  = [240, 240, 240, 240, 240, 240, 240, 500, 960]

def sail():
	freq = [659, 659, 659, 784, 784, 784, 880, 987, 880, 784, 659, 165]
	dur  = [130, 130, 130, 130, 130, 130, 400, 400, 300, 130, 400, 700]

	play(freq, dur)

def music(sound):
	winsound.PlaySound("%s.wav" %sound, winsound.SND_FILENAME)

def play(freq, dur):
	for i in range(0,len(freq)):
		winsound.Beep(freq[i], dur[i])
# just in case you wanna run straight from command line
#if __name__ == "__main__":
	#music(sys.argv[1])