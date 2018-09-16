import subprocess

variables = ['brain1.tiff', 'brain2.tiff', 'prostata.tiff', 'cells.png', 'nyc4k.jpg']

program = '../Implementations/C++/bin/./FeatureExtractor'
optionWindowSize = '-w'
optionWindowSizeValue = '3'
optionSymmtrecity = '-g'
optionInputFile = '-i'  
optionSave = ''

for variable in variables:	
    
    result = subprocess.run([program, '-i', variable, optionWindowSize, optionWindowSizeValue, optionSave], stdout=subprocess.PIPE)
    print(subprocess.list2cmdline(result.args))

    output = result.stdout.decode('utf-8').strip()
    print(output)

    with open('CPU_' + variable +'_RESULT.txt', 'w') as file:
        file.write(output)


