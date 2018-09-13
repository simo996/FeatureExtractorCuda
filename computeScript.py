import subprocess

variables = ['brain1.tiff', 'brain2.tiff']

optionWindowSize = '-w'
optionSymmtrecity = '-g'
optionInputFile = '-i'  

for variable in variables:
    
    result = subprocess.run(['./FeatureExtractor', '-i', variable, optionWindowSize, '7'], stdout=subprocess.PIPE)
    print(subprocess.list2cmdline(result.args))

    output = result.stdout.decode('utf-8').strip()
    print(output)

    with open(variable +'_RESULT.txt', 'w') as file:
        file.write(output)

