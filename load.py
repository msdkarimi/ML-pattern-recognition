import numpy
import util

class Load :
    def __init__(self, address):
        self.address = address
        self.samples, self.labels = self.extractData()
        self.prior = 0.1


    def extractData(self):
        mapping = {
            "0": 0,
            "1": 1
        }
        try:
            with open(self.address, "r") as file:
                theFile = [eachLine.strip().split(",") for eachLine in file]

            labelsInString = [ eachSample[-1]  for eachSample in theFile]
            features = [util.toCol(numpy.array(eachSample[0: len(eachSample)-1],dtype=float)) for eachSample in theFile]

            mappedLabels= []
            for label in labelsInString:
                mappedLabels.append(mapping[label])

            return numpy.hstack(features), numpy.array(mappedLabels, dtype=int)

        except IOError:
            print(f"There is a problem with opening {self.address} file.")
