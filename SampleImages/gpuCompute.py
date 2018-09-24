import subprocess

images = ['brain', 'prostate', 'uterine', 'fluorescence']
windowSizes = [3, 5, 9, 15, 23, 33]

program = '../Implementations/Cuda/bin/./Cufeat'
optionWindowSize = '-w'
optionSymmtrecity = '-g'
optionInputFile = '-i'  
optionSave = '-s'

for image in images:	
	
	for i in range(1, 15):
		
		actualImage = image + str(i) + '.tiff'

		for optionWindowSizeValue in windowSizes:

			result = subprocess.run([program, '-i', actualImage, optionWindowSize, str(optionWindowSizeValue), optionSave], stdout=subprocess.PIPE)
			print(subprocess.list2cmdline(result.args))

			output = result.stdout.decode('utf-8').strip()
			print(output)

			with open('GPU_' + actualImage + '_WSIZE=' + str(optionWindowSizeValue) +'_RESULT.txt', 'w') as file:
				file.write(output)