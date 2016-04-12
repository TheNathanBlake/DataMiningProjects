import winsound

# You used to debug on Ubuntu
def drake():
	freq = [659, 659, 659, 698, 659, 587, 523, 659, 523]
	dur  = [240, 240, 240, 240, 240, 240, 240, 500, 960]

	for i in range(0,len(freq)):
		winsound.Beep(freq[i], dur[i])

def music(sound):
	winsound.PlaySound("%s.wav" %sound, winsound.SND_FILENAME)
	
# just in case you wanna run straight from command line
#if __name__ == "__main__":
	#music(sys.argv[1])