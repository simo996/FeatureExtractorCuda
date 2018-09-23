import subprocess

images = ['brain1.tiff', 'brain2.tiff', 'prostata.tiff']
windowSizes = [3, 5, 7, 9, 11, 13, 15]

program = '../Implementations/Cuda/bin/./CuFeat'
optionWindowSize = '-w'
optionSymmtrecity = '-g'
optionInputFile = '-i'  
optionSave = '-s'

for image in images:	
    
    for optionWindowSizeValue in windowSizes:

	    result = subprocess.run([program, '-i', image, optionWindowSize, str(optionWindowSizeValue), optionSave], stdout=subprocess.PIPE)
	    print(subprocess.list2cmdline(result.args))

	    output = result.stdout.decode('utf-8').strip()
	    print(output)

	    with open('GPU_' + image + '_WSIZE=' + str(optionWindowSizeValue) +'_RESULT.txt', 'w') as file:
	        file.write(output)

