import subprocess

variables = ['brain1.tiff', 'brain2.tiff', 'prostata.tiff', 'cells.png', 'nyc4k.jpg']

program = '../Implementations/Pre-Cuda/bin/./FeatureExtractor'
optionWindowSize = '-w'
optionSymmtrecity = '-g'
optionInputFile = '-i'  
optionSave = ''

for variable in variables:	
    
    result = subprocess.run([program, '-i', variable, optionWindowSize, '3', optionSave], stdout=subprocess.PIPE)
    print(subprocess.list2cmdline(result.args))

    output = result.stdout.decode('utf-8').strip()
    print(output)

    with open('CPU_' + variable +'_RESULT.txt', 'w') as file:
        file.write(output)


