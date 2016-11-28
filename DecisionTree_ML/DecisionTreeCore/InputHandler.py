import os.path

class InputHandler:
    """
    This is InputHandler class responsible for reading inputs to the program
    """

    def convertToInteger(self, numberString):
        """
        This function checks if input is number and if yes then convert it into integer.
        :param numberString:
            Number string to convert into integer
        :return:
            Converted integer if string is digit else the same string
        """
        try:
            numberString = float(numberString)
            numberString = int(numberString)
        except ValueError:
            pass

        return numberString


    def readFile(self, inputFile, labelIndex, featureIndices=None, fileType="text", ignoreHeader=False):
        """
        This function reads the input file to the program and generate the data matrix for the program.

        :param inputFile: path to input file
        :param labelIndex: index of class label column in the data
        :param featureIndices: indices of the feature columns in the data
        :param fileType: type of file to read (text/csv)
        :param ignoreHeader: ignores first line of file if set to True
        :return: data matrix with in format [[feature1, feature2, .., classlabel],..]
        """
        if os.path.isfile(inputFile) == False:
            print('Invalid file path')
            return 0
        else:
            with open(inputFile) as f:
                # ignore header if set to true
                if ignoreHeader:
                    next(f)

                featureMatrix = []
                for line in f:
                    lineData = [ self.convertToInteger(x) for x in line.rstrip('\n').split(self.getSplitChar(fileType))]
                    label = lineData[labelIndex]
                    if featureIndices is not None:
                        features = lineData[featureIndices[0]:featureIndices[1]]
                    else:
                        features = [x for i,x in enumerate(lineData) if i!=labelIndex and i!=labelIndex+1]  # for mushroom dataset, ignore 'bruises?-no' feature as well
                    features.append(label)
                    featureMatrix.append(features)
            return featureMatrix


    def getSplitChar(self, fileType):
        """
        This function returns either a comma (,) or a space ( )
        which is used as a split character based on file-type
        :param fileType:
        :return:
        """
        if fileType == "csv":
            return ','
        else:
            return ' '