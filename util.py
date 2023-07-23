import array
import math
import pickle

from tqdm import tqdm
import os
import numpy
import scipy

def toCol(theArray:numpy.array):
    try:
        return theArray.reshape(theArray.size, 1)
    except Exception as e:
        print(e)

def toRow(theArray:numpy.array):
    try:
        return theArray.reshape(1,theArray.size)
    except Exception as e:
        print(e)

def mean(samples: numpy.array):
    try:
        return samples.mean(1).reshape(samples.shape[0], 1)
    except Exception as e:
        print(e)


def centerSampels (samples: numpy.array):
    try:
        mean = samples.mean(1).reshape(samples.shape[0], 1)
        return samples - mean

    except Exception as e:
        print(e)

def covarinceMartix( sampels:numpy.array )-> numpy.array:
    try:
        return   numpy.dot(centerSampels(sampels), centerSampels(sampels).T)/ sampels.shape[1]

    except Exception as e:
        print(e)

#
# def kfold(samples: numpy.array, labels, foldNumber: int, folds: int = 3, seed = 0):
#     # maskOfEachClass = dict()
#     # for classLabel in range(len(set(labels))):
#     #     maskOfGivenClass = numpy.where (labels == classLabel)
#     #     maskOfEachClass[classLabel] = samples[:, maskOfGivenClass[0]]
#     numpy.random.seed(seed)
#     idx = numpy.random.permutation(samples.shape[1])
#
#     testDataIndex = [foldNumber]
#     trainDataIndx = list(set(idx) - set(testDataIndex))
#
#     testData = samples[:, testDataIndex]
#     testLabel = labels[testDataIndex]
#
#     trainData = samples[:, trainDataIndx]
#     trainLabel = labels[trainDataIndx]
#
#     return (trainData, trainLabel), (testData, testLabel)

def PCA(samples: numpy.array, covarianceMatrix: numpy.array, m: int = 4)->tuple:

    try:
        U, s, _ = numpy.linalg.svd(covarianceMatrix)
        P = U[:, 0:m]
        mapped = numpy.dot(P.T, samples)
        return mapped, P

    except Exception as e:
        print(e)

def LDA(samples: numpy.array, labels: numpy.array, m: int = 2):

    try:
        sW, sB = getSW_SB(samples, labels)
        _, U = scipy.linalg.eigh(sB, sW)
        W = U[:, ::-1][:, 0:m]

        mapped = numpy.dot(W.T, samples)
        return mapped, W

    except Exception as e:
        print(e)


def getSW_SB(dataSet, labels):
    try:
        sW = 0
        for i in range(len(set(labels))):
            eachClass = dataSet[:, labels == i]
            nEachClass = eachClass.shape[1]
            swCTemp = covarinceMartix(eachClass)
            swCTemp = swCTemp * nEachClass
            sW = sW + swCTemp

        sW = sW / dataSet.shape[1]

        sB = 0
        for i in range(len(set(labels))):
            mu = mean(dataSet)
            eachClass = dataSet[:, labels == i]
            nEachClass = eachClass.shape[1]
            muC = mean(eachClass)
            sBtemp = nEachClass * numpy.dot((muC - mu), (muC - mu).T)

            sB = sB + sBtemp

        sB = sB / dataSet.shape[1]

        return sW, sB
    except Exception as e:
        print(e)



def logpdf_GAU_ND(X, mu, C):
    try:
        y = [ logpdfOneSample(X[:,i:i+1], mu, C) for i in range(X.shape[1])]
        return numpy.array(y).ravel()

    except Exception as e:
        print(e)

def logpdfOneSample(x, mu, C):
    try:
        xc = x - mu
        M = x.shape[0]
        constant = -0.5 * M * numpy.log(2*numpy.pi)
        logDetSigma = numpy.linalg.slogdet(C)[1]
        invSigma = numpy.linalg.inv(C)
        vector = numpy.dot(xc.T, numpy.dot(invSigma,xc))
        return constant - 0.5 * logDetSigma - 0.5 * vector

    except Exception as e:
        print(e)
def preprocessing(Xtrain, Xtest, pca=False, zNorm=False, pcaM = 12):
    trainData = Xtrain
    testData = Xtest
    if zNorm and not pca:
        pp = Preprocess(trainSet=Xtrain, testSet=Xtest)
        pp.znorm()
        trainData = pp.zNormTrain
        testData = pp.zNormTest

    elif not zNorm and pca:
        # print("just pca")
        pp = Preprocess(trainSet=Xtrain, testSet=Xtest, pcaM=pcaM)
        pp.preprocess_PCA()
        trainData = pp.PCAmappedTrain
        testData = pp.PCAmappedTest

    # if self.gaussianization and self.pca:
    elif zNorm and pca:
        # print("both")
        ppZ = Preprocess(trainSet=Xtrain, testSet=Xtest)
        ppZ.znorm()
        ppPCA = Preprocess(trainSet=ppZ.zNormTrain, testSet=ppZ.zNormTest, pcaM=pcaM)
        ppPCA.preprocess_PCA()
        trainData = ppPCA.PCAmappedTrain
        testData = ppPCA.PCAmappedTest

    return trainData, testData

class WorkingPoint:
    def __init__(self, target_prior = 0.5 , fnC=1.0, fpC=1.0, errorPlot=False):
        self.target_prior = target_prior
        self.fnC = fnC
        self.fpC = fpC
        if not errorPlot:
            self.piTilde = (target_prior * fnC) / (target_prior * fnC + ((1 - target_prior) * fpC))
        else:
            self.piTilde = 1 / (1 + numpy.exp(-1*target_prior))
        self.effectiveThresshold = -1 * numpy.log(self.piTilde/(1-self.piTilde))


class MVG:
    def __init__(self, kfold, model = "G", zNorm=False, pca = False, pcaM = 12, tqdm=True):
        self.allFolds = kfold
        self.model = model
        self.kfoldLoglr = list()
        self.kfoldLabels = list()
        self.tqdm = not tqdm

        self.pca = pca
        self.zNorm = zNorm
        self.pcaM = pcaM

        self.train_validation()



    def train_validation(self):
        for (Xtrain, Ytrain), (Xtest, Ytest) in tqdm(self.allFolds, disable=self.tqdm):
            trainData, testData = preprocessing(Xtrain=Xtrain, Xtest=Xtest, zNorm=self.zNorm, pca=self.pca, pcaM=self.pcaM)
            trainLabel = Ytrain
            testLabel = Ytest

            # pp = Preprocess(trainSet=, testSet=)
            # print(testLabel.shape)

            means = {}
            sigmas = {}
            tiedSigma = 0

            for i in range(len(set(trainLabel))):
                theClassOfLabel = trainData[:, trainLabel == i]
                mu = mean(theClassOfLabel)
                sigma = covarinceMartix(theClassOfLabel)
                if self.model == "N":
                    I = numpy.eye(trainData.shape[0], trainData.shape[0])
                    sigma = sigma * I
                means[i] = mu
                sigmas[i] = sigma

            if self.model == "T":
                for i in range(len(set(trainLabel))):
                    theSigma = sigmas[i]
                    givenClassSize = trainData[:, trainLabel == i].shape[1]
                    tiedSigma += (givenClassSize * theSigma)

                tiedSigma /= trainData.shape[1]

            logLiklihood = numpy.zeros((len(set(testLabel)), testData.shape[1]))
            for i in range(len(set(testLabel))):
                muML = means[i]
                if self.model != "T":
                    sigmaML = sigmas[i]
                else:
                    sigmaML = tiedSigma
                likelihhod = logpdf_GAU_ND(testData, muML, sigmaML)
                logLiklihood[i, :] = likelihhod

            self.kfoldLoglr.append(logLiklihood[1, :] - logLiklihood[0, :])
            self.kfoldLabels.append(testLabel)

        self.logLiklihoodRatio = numpy.hstack(self.kfoldLoglr)
        self.groundTruthLabels = numpy.hstack(self.kfoldLabels)

class KFold:
    def __init__(self, samples, labels, n_splits=3, seed=0):
        self.numberOfFolds = n_splits
        self.seed = seed
        self.samples = samples
        self.labels = labels
        self.kfSplite()


    def kfSplite(self):

        Xdataset = self.samples
        Ydataset = self.labels

        pairOfTrainTest = list()
        numpy.random.seed(self.seed)
        indices = numpy.random.permutation(Xdataset.shape[1])
        folds = numpy.array_split(indices, self.numberOfFolds)
        ranges = set(range(self.numberOfFolds))
        for fold in range(self.numberOfFolds):
            test_indices = numpy.hstack(folds[fold])
            train_indices = numpy.hstack(numpy.concatenate(folds[:fold] + folds[fold + 1:]))

            Xtrain = Xdataset[:, train_indices]
            Ytrain = Ydataset[train_indices]

            Xtest = Xdataset[:, test_indices]
            Ytest = Ydataset[test_indices]

            pairOfTrainTest.append(((Xtrain, Ytrain),(Xtest, Ytest)))
        self.allFolds = pairOfTrainTest


class LogisticRegression:
    def __init__(self, kfold, workingPoint, lambdaa=1e-5, pca=False, zNorm=False, pcaM=12, tqdm = True, save=True):
        self.dir = 'result/lr/'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.dir_pca = 'result/lr/pca/'
        if not os.path.exists(self.dir_pca):
            os.makedirs(self.dir_pca)
        self.allFolds = kfold
        self.kfoldLoglr = list()
        self.kfoldLabels = list()
        self.tqdm = not tqdm

        self.lambdaa = lambdaa
        self.wp = workingPoint
        self.save=save
        self.pca = pca
        self.zNorm = zNorm
        self.pcaM = pcaM

        self.train_validation()

    def train_validation(self):
        for (Xtrain, Ytrain), (Xtest, Ytest) in tqdm(self.allFolds, disable= self.tqdm):
            trainData, testData = preprocessing(Xtrain=Xtrain, Xtest=Xtest, zNorm=self.zNorm, pca=self.pca, pcaM=self.pcaM)
            trainLabel = Ytrain
            testLabel = Ytest

            scores, predictedLabels = self.train_logReg_prior_weighted(trainData, trainLabel, testData, priorTarget = self.wp.piTilde, lambdaa=self.lambdaa)
            self.kfoldLabels.append(testLabel)
            self.kfoldLoglr.append(scores - numpy.log(self.wp.piTilde/(1-self.wp.piTilde)))

        self.logLiklihoodRatio = numpy.hstack(self.kfoldLoglr)[0]
        self.groundTruthLabels = numpy.hstack(self.kfoldLabels)
        if self.save:
            fileName = "lr_gamma_" + str(self.lambdaa) + "_pca_" + str(self.pca) + "_" + str(
                self.pcaM) + "_" + "zNorm_" + str(self.zNorm)+"_piThilde_"+str(self.wp.piTilde)
            labelFileName = "label_lr_gamma_" + str(self.lambdaa) + "_pca_" + str(self.pca) + "_" + str(
                self.pcaM) + "_" + "zNorm_" + str(self.zNorm)+"_piThilde_"+str(self.wp.piTilde)
            if not self.pca:
                with open(self.dir + fileName, 'wb') as f:
                    pickle.dump(self.logLiklihoodRatio, f)
                with open(self.dir + labelFileName, 'wb') as f:
                    pickle.dump(self.groundTruthLabels, f)
            else:
                with open(self.dir_pca + fileName, 'wb') as f:
                    pickle.dump(self.logLiklihoodRatio, f)
                with open(self.dir_pca + labelFileName, 'wb') as f:
                    pickle.dump(self.groundTruthLabels, f)

    def logReg_obj_wrap_prior_weighted(self, trainData, trainLabel, priorTarget, lambdaa):
        features = trainData.shape[0]
        Z = trainLabel * 2.0 - 1.0

        class0_size = len(trainLabel[trainLabel == 0])
        class1_size = len(trainLabel[trainLabel == 1])

        class0_z = Z[trainLabel == 0]
        class1_z = Z[trainLabel == 1]

        def logReg(V):
            w = toCol(V[0:features])
            b = V[-1]

            class0_scores = numpy.dot(w.T, trainData[:, trainLabel == 0]) + b
            class1_scores = numpy.dot(w.T, trainData[:, trainLabel == 1]) + b

            class0_prior_Weight_loss = ((1-priorTarget)/class0_size) * numpy.logaddexp(0, -class0_z * class0_scores).sum()
            class1_prior_Weight_loss = ((priorTarget)/class1_size) * numpy.logaddexp(0, -class1_z * class1_scores).sum()
            regulizer = 0.5 * lambdaa * numpy.linalg.norm(w)**2

            return regulizer + class1_prior_Weight_loss + class0_prior_Weight_loss

            # S = numpy.dot(w.T, trainData) + b
            # cxe = numpy.logaddexp(0, -S * Z).mean()
            # return cxe + 0.5 * lambdaa * numpy.linalg.norm(w) ** 2
        return logReg


    def train_logReg_prior_weighted(self, trainData, trainLabel, testData, priorTarget = 0.5, lambdaa=0.001):
        logReg_obj = self.logReg_obj_wrap_prior_weighted(trainData, trainLabel, priorTarget, lambdaa)
        x0 = numpy.zeros(trainData.shape[0]+1)
        V, _, _ = scipy.optimize.fmin_l_bfgs_b(logReg_obj, x0=x0, approx_grad=True)
        w, b = toCol(V[0:trainData.shape[0]]), V[-1]
        scores = numpy.dot(w.T, testData) + b
        predictedLabels = (scores.ravel() > 0) * 1
        return scores, predictedLabels

class MeasuringPredictions:
    def __init__(self, ground_truth_labels, predicted_labels, fnC, fpC, target_prior = 0.5):

        self.predicted_labels = predicted_labels
        self.ground_truth_labels = ground_truth_labels
        self.targetPrior = target_prior
        self.fpC = fpC
        self.fnC = fnC
        self.confusion_matrix()
        self.normal_dcf()

    def confusion_matrix(self):
        ground_truth_labels = self.ground_truth_labels
        predicted_labels = self.predicted_labels
        self.cm = numpy.zeros((len(set(ground_truth_labels)), len(set(ground_truth_labels))))

        mask_GT = ground_truth_labels == 0
        predicted = predicted_labels[mask_GT]
        ground_truth = ground_truth_labels[mask_GT]
        tN = sum(predicted == ground_truth)
        fP = sum (ground_truth_labels == 0) - tN

        mask_GT = ground_truth_labels == 1
        predicted = predicted_labels[mask_GT]
        ground_truth = ground_truth_labels[mask_GT]
        tP = sum(predicted == ground_truth)
        fN = sum(ground_truth_labels == 1) - tP

        self.cm[0][0] = tN
        self.cm[0][1] = fP
        self.cm[1][0] = fN
        self.cm[1][1] = tP
        self.tpR = tP/(fN + tP)
        self.fpR = fP/(fP+tN)
        self.fnR = 1 - self.tpR
        self.tnR = 1 - self.fpR
        # return cm, tpR, fpR, fnR, tnR

    def empirical_bayes_risk(self):
        return (self.targetPrior * self.fnC * self.fnR) + ( (1-self.targetPrior)* self.fpC * self.fpR)

    def normal_dcf(self):
        self.normalDCF = self.empirical_bayes_risk()/ min(self.targetPrior*self.fnC, (1- self.targetPrior)*self.fpC)


class Preprocess:
    def __init__(self, trainSet, testSet, pcaM = 12):
        self.trainSet = trainSet
        self.testSet = testSet
        # self.gaussianiz_trainSet()
        # self.gaussianiz_testSet()

        self.pcaM = pcaM

    def gaussianiz_trainSet(self):
        gauss_DTR = numpy.zeros(self.trainSet.shape)
        for f in range(self.trainSet.shape[0]):
            gauss_DTR[f, :] = scipy.stats.norm.ppf(scipy.stats.rankdata(self.trainSet[f, :], method="min") / (self.trainSet.shape[1] + 2))
        self.gaussnizedTrainSet =  gauss_DTR


    def gaussianiz_testSet(self):
        gauss_DTE = numpy.zeros(self.testSet.shape)
        for f in range(self.trainSet.shape[0]):
            for idx, x in enumerate(self.testSet[f, :]):
                rank = 0
                for x_i in self.trainSet[f, :]:
                    if (x_i < x):
                        rank += 1
                uniform = (rank + 1) / (self.trainSet.shape[1] + 2)
                gauss_DTE[f][idx] = scipy.stats.norm.ppf(uniform)

        self.gaussizedTestSet = gauss_DTE

    def preprocess_PCA(self):
        C = covarinceMartix(self.trainSet)
        mappedTrain, P = PCA( self.trainSet, C, self.pcaM)
        self.PCAmappedTrain = mappedTrain
        self.PCAmappedTest  = numpy.dot(P.T, self.testSet)
        # return mappedTrain, mappedTest

    def znorm(self):
        mu_DTR = toCol(mean(self.trainSet))
        std_DTR = toCol(numpy.std(self.trainSet))
        # print(std_DTR)
        # print(mu_DTR)
        # exit(0)

        self.zNormTrain = (self.trainSet - mu_DTR) / std_DTR
        self.zNormTest= (self.testSet - mu_DTR) / std_DTR



def min_DCF(llr, groundTruthLabels, workingPoint):
    llr = llr.ravel()
    tresholds = numpy.concatenate([numpy.array([-numpy.inf]),numpy.sort(llr),numpy.array([numpy.inf])])
    DCF = numpy.zeros(tresholds.shape[0])
    for (idx, t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        mp = MeasuringPredictions(ground_truth_labels=groundTruthLabels, predicted_labels=pred, fnC=1.0, fpC=1.0, target_prior=workingPoint.piTilde)
        DCF[idx] = mp.normalDCF
    argmin = DCF.argmin()
    return DCF[argmin], tresholds[argmin]


def computeCorrelationMatrix(fullFeatureMatrix):
    C = covarinceMartix(fullFeatureMatrix)
    correlations = numpy.zeros((C.shape[1], C.shape[1]))
    for x in range(C.shape[1]):
        for y in range(C.shape[1]):
            correlations[x,y] = numpy.abs( C[x,y] / ( numpy.sqrt(C[x,x]) * numpy.sqrt(C[y,y]) ) )
    return correlations

# def SVM_RadialBasisKernel_model(data, labels, gamma=1.0, C=1.0, K=1):
#     data = numpy.vstack([data, numpy.ones((1, data.shape[1]))])
#     Z = numpy.zeros(labels.shape)
#     Z[labels == 1] = 1
#     Z[labels == 0] = -1
#     Khat = numpy.zeros((data.shape[1], data.shape[1]))
#     for i in range(data.shape[1]):
#         for j in range(data.shape[1]):
#             Khat[i, j] = RadialBasisKernel(data[:, i], data[:, j], gamma, K)
#     H = toCol(Z) * toRow(Z) * Khat
#
#     def JDual(alpha):
#         temp = numpy.dot(H, toCol(alpha))
#         Jhat = -1 * 0.5 * numpy.dot(toRow(alpha), temp).ravel()
#         Jhat += alpha.sum()
#         grad = numpy.ones(alpha.size) - temp.ravel()
#         return Jhat, grad
#
#     def LDual(alpha):
#         loss, grad = JDual(alpha)
#         return -1*loss, -1*grad
#
#     alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(LDual, numpy.zeros(data.shape[1]),bounds=[(0, C)] * data.shape[1], factr=1.0, maxiter=100000, maxfun=100000)
#     return alphaStar
#
#
# def SVM_RadialBasisKernel_classification(trainingData, trainingLabels, testData, alphaStar, gamma=1.0, K=1):
#     trainingData = numpy.vstack([trainingData, numpy.ones((1, trainingData.shape[1]))])
#     testData = numpy.vstack([testData, numpy.ones((1, testData.shape[1]))])
#     Z = numpy.zeros(trainingLabels.shape)
#     Z[trainingLabels == 1] = 1
#     Z[trainingLabels == 0] = -1
#     score = numpy.zeros((testData.shape[1], ))
#
#     for l in range(testData.shape[1]):
#         for i in range(trainingData.shape[1]):
#             if alphaStar[i] > 0:
#                 score[l] += Z[i] * alphaStar[i] * \
#                     RadialBasisKernel(trainingData[:, i], testData[:, l], gamma, K)
#
#     return score
#
# def svm(allFolds, gamma =1.0, C=1.0):
#     scores = list()
#     labels = list()
#     for (Xtrain, Ytrain), (Xtest, Ytest) in tqdm(allFolds):
#         alphaStar = SVM_RadialBasisKernel_model(Xtrain, Ytrain, gamma=gamma, C=C)
#         score = SVM_RadialBasisKernel_classification(Xtrain, Ytrain, Xtest, alphaStar, gamma =1.0)
#         labels.append(Ytest)
#         scores.append(score)
#     return numpy.hstack(scores), numpy.hstack(labels)

class SupportVectorMachine:
    def __init__(self, kfold, targetPrior=None, rbf=False, polynomial=False, C=1.0, gamma=1.0, polynomial_c=0.0, d=2, pca=False, zNorm=False, pcaM=10, tqdm=True, eval=False):
        self.dir = 'result/svm/'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if eval:
            self.dirEval = 'result/svm/eval/'
            if not os.path.exists(self.dirEval):
                os.makedirs(self.dirEval)

        self.allFolds = kfold
        self.rbf = rbf
        self.polynomial = polynomial
        self.C = C
        self.polynomial_c = polynomial_c
        self.d = d
        self.tqdm = not tqdm
        self.targetPrior = targetPrior
        self.gamma = gamma
        self.scores = list()
        self.groundTrouthLabels = list()
        self.eval= eval
        self.pca = pca
        self.zNorm = zNorm
        self.pcaM = pcaM

        self.train()

    def train(self):
        for (Xtrain, Ytrain), (Xtest, Ytest) in tqdm(self.allFolds, disable=self.tqdm):
            trainData, testData = preprocessing(Xtrain=Xtrain, Xtest=Xtest, zNorm=self.zNorm, pca=self.pca, pcaM=self.pcaM)
            self.testSet = testData
            self.testSetLabel = Ytest

            self.testSetExtended = numpy.vstack([self.testSet, numpy.ones((1, self.testSet.shape[1]))])

            self.trainSetExtended = numpy.vstack([trainData, numpy.ones((1, Xtrain.shape[1]))])
            self.trainSetLabel = Ytrain

            Z = numpy.zeros(self.trainSetLabel.shape)
            Z[self.trainSetLabel == 1] = 1
            Z[self.trainSetLabel == 0] = -1

            if not self.rbf and not self.polynomial:
                # print("linear")
                H = numpy.dot(self.trainSetExtended.T, self.trainSetExtended)
            else:
                if self.rbf:
                    H = self.rbf_kernel()
                else:
                    # print("poly")
                    H = self.polynominal_kernel()

            H = toCol(Z) * toRow(Z) * H

            def JDual(alpha):
                Ha = numpy.dot(H, toCol(alpha))
                aHa = numpy.dot(toRow(alpha), Ha)
                a1 = alpha.sum()
                return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

            def lDual(alpha):
                loss, gradiant = JDual(alpha)
                return -loss, -gradiant

            def JPrimal(w):
                S = numpy.dot(toRow(w), self.trainSetExtended)
                loss = numpy.maximum(numpy.zeros(S.shape), 1 - Z * S).sum()
                return 0.5 * numpy.linalg.norm(w) ** 2 + self.C * loss

            if self.targetPrior is None:
                alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                    lDual,
                    numpy.zeros((self.trainSetExtended.shape[1])),
                    bounds=[(0, self.C)] * self.trainSetExtended.shape[1],
                    factr=1.0,
                    maxiter=100000,
                    maxfun=100000)
            else:
                mask = self.trainSetLabel == 1
                c1 = self.C * self.trainSetExtended.shape[1] * (self.targetPrior/self.trainSetExtended[:, self.trainSetLabel == 1].shape[1])
                c0 = self.C * self.trainSetExtended.shape[1] * ((1-self.targetPrior)/self.trainSetExtended[:, self.trainSetLabel == 0].shape[1])

                alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                    lDual,
                    numpy.zeros((self.trainSetExtended.shape[1])),
                    bounds= [(0, c1) if m else (0, c0) for m in mask],
                    factr=1.0,
                    maxiter=100000,
                    maxfun=100000)

            wStar = numpy.dot(self.trainSetExtended, toCol(alphaStar) * toCol(Z))
            if not self.rbf and not self.polynomial:
                self.wStar = wStar
            else:
                self.alphaStar = toRow(alphaStar)

            self.score_test_data()

        self.allScores = numpy.hstack(self.scores)[0]
        self.groundTrouthLabels = numpy.hstack(self.groundTrouthLabels)
        fileName= "svm_rbf"+str(self.rbf)+"_gamma_"+str(self.gamma)+"_poly_"+ str(self.polynomial)+"_"+"pca_"+str(self.pca)+"_"+str(self.pcaM)+"_"+"zNorm_"+str(self.zNorm)+"_"+"balanced_"+str(self.targetPrior)
        fileNameLabel= "label_svm_rbf"+str(self.rbf)+"_gamma_"+str(self.gamma)+"_poly_"+ str(self.polynomial)+"_"+"pca_"+str(self.pca)+"_"+str(self.pcaM)+"_"+"zNorm_"+str(self.zNorm)+"_"+"balanced_"+str(self.targetPrior)
        if not self.eval:
            with open(self.dir+fileName, 'wb') as f:
                pickle.dump(self.allScores, f)
            with open(self.dir+fileNameLabel, 'wb') as f:
                pickle.dump(self.groundTrouthLabels, f)
        else:
            with open(self.dirEval + fileName, 'wb') as f:
                pickle.dump(self.allScores, f)
            with open(self.dirEval + fileNameLabel, 'wb') as f:
                pickle.dump(self.groundTrouthLabels, f)


    def score_test_data(self):
        # self.testSetExtended = numpy.vstack([self.testSet, numpy.ones((1, self.testSet.shape[1]))])
        if not self.rbf and not self.polynomial:
            self.scores.append(numpy.dot(self.wStar.T, self.testSetExtended))
            self.groundTrouthLabels.append(self.testSetLabel)
        else:
            Z = numpy.zeros(self.trainSetLabel.shape)
            Z[self.trainSetLabel == 1] = 1
            Z[self.trainSetLabel == 0] = -1
            if self.rbf:
                rbfKernel = self.rbf_kernel(train=False)
                self.scores.append(numpy.dot(self.alphaStar * Z, rbfKernel))
                # self.scores.append(score)
                self.groundTrouthLabels.append(self.testSetLabel)
            else:
                ployKernel = self.polynominal_kernel(train=False)
                self.scores.append(numpy.dot(self.alphaStar * Z, ployKernel))
                # self.scores.append(score)
                self.groundTrouthLabels.append(self.testSetLabel)

    def rbf_kernel(self, train=True):
        # self.testSetExtended = numpy.vstack([self.testSet, numpy.ones((1, self.testSet.shape[1]))])
        if train:
            dis = toCol((self.trainSetExtended ** 2).sum(0)) + toRow((self.trainSetExtended ** 2).sum(0)) - 2 * numpy.dot(self.trainSetExtended.T, self.trainSetExtended)
            rbfKernel = numpy.exp(-self.gamma * dis)
            return rbfKernel
        else:
            dis = toCol((self.trainSetExtended ** 2).sum(0)) + toRow((self.testSetExtended ** 2).sum(0)) - 2 * numpy.dot(self.trainSetExtended.T, self.testSetExtended)
            rbfKernel = numpy.exp(-self.gamma * dis)
            return rbfKernel

    def polynominal_kernel(self, train=True):
        if train:
            return (numpy.dot(self.trainSetExtended.T, self.trainSetExtended) + self.polynomial_c) ** self.d
        else:
            return (numpy.dot(self.trainSetExtended.T, self.testSetExtended) + self.polynomial_c) ** self.d


def RadialBasisKernel(x1, x2, gamma, K=1):
    result = x1 - x2
    result = -1 * numpy.linalg.norm(result) ** 2
    result = numpy.exp(gamma * result)
    # result += K**2
    return result

def polynomial_kernel(x1, x2, c, d, K=1):
    result = numpy.dot(x1.T, x2) + c
    result **= d
    # result += K**2
    return result
# down old one
# class SupportVectorMachine:
#     def __init__(self, kfold, targetPrior=None, rbf=False, polynomial=False, c=1.0, gamma=1.0, tqdm = True):
#         self.allFolds = kfold
#         self.rbf = rbf
#         self.polynomial = polynomial
#         self.c = c
#         self.targetPrior = targetPrior
#
#         self.tqdm = not tqdm
#
#         self.gamma = gamma
#         self.scores = list()
#         self.groundTrouthLabels = list()
#
#         self.train()
#
#
#
#
#     def train(self):
#
#         for (Xtrain, Ytrain), (Xtest, Ytest) in tqdm(self.allFolds, disable=self.tqdm):
#             self.testSet = Xtest
#             self.testSetLabel = Ytest
#
#
#             if not self.rbf and not self.polynomial:
#                 trainSetExtended = numpy.vstack([Xtrain, numpy.ones((1, Xtrain.shape[1]))])
#                 Z = numpy.zeros(Ytrain.shape)
#                 Z[Ytrain == 1] = 1
#                 Z[Ytrain == 0] = -1
#                 H = numpy.dot(trainSetExtended.T, trainSetExtended)
#                 H = toCol(Z) * toRow(Z) * H
#
#             else:
#                 pass
#
#             def JDual(alpha):
#                 Ha = numpy.dot(H, toCol(alpha))
#                 aHa = numpy.dot(toRow(alpha), Ha)
#                 a1 = alpha.sum()
#                 return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)
#
#             def lDual(alpha):
#                 loss, gradiant = JDual(alpha)
#                 return -loss, -gradiant
#
#             def JPrimal(w):
#                 S = numpy.dot(toRow(w), trainSetExtended)
#                 loss = numpy.maximum(numpy.zeros(S.shape), 1 - Z * S).sum()
#                 return 0.5 * numpy.linalg.norm(w) ** 2 + self.c * loss
#
#             if self.targetPrior is None:
#                 alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
#                     lDual,
#                     numpy.zeros((Xtrain.shape[1])),
#                     bounds=[(0, self.c)] * Xtrain.shape[1],
#                     factr=1.0,
#                     maxiter=100000,
#                     maxfun=100000)
#             else:
#                 mask = Ytrain == 1
#                 # print(mask.shape)
#                 # exit(0)
#                 # print(mask)
#                 # print(Ytrain)
#                 # empericlPrior = trueClass/allSamples
#                 c1 = (self.c * Xtrain.shape[1] * self.targetPrior) / (Xtrain[:, Ytrain == 1].shape[1])
#                 c0 = (self.c * Xtrain.shape[1] * (1-self.targetPrior)) / (Xtrain[:, Ytrain == 0].shape[1])
#                 # print([(0, c1) if m else (0, c0) for m in mask])
#                 # exit(0)
#
#
#                 alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
#                     lDual,
#                     numpy.zeros((Xtrain.shape[1])),
#                     bounds= [(0, c1) if m else (0, c0) for m in mask],
#                     factr=1.0,
#                     maxiter=100000,
#                     maxfun=100000)
#
#
#             wStar = numpy.dot(trainSetExtended, toCol(alphaStar) * toCol(Z))
#             if not self.rbf and not self.polynomial:
#                 self.wStar = wStar
#             else:
#                 self.alphaStar = toRow(alphaStar)
#
#             self.score_test_data()
#
#         self.allScores = numpy.hstack(self.scores)[0]
#         self.groundTrouthLabels = numpy.hstack(self.groundTrouthLabels)
#
#     def score_test_data(self):
#         if not self.rbf and not self.polynomial:
#             testSetExtended = numpy.vstack([self.testSet, numpy.ones((1, self.testSet.shape[1]))])
#             self.scores.append(numpy.dot(self.wStar.T, testSetExtended))
#             self.groundTrouthLabels.append(self.testSetLabel)

def GMM_EM(X,GMM, thresholdForEValues, model):
    llNew = None
    llOld = None
    G = len(GMM)
    N = X.shape[1]
    gmmnew = None
    while llOld is None or (llNew - llOld) > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G,N))
        for g in range(G):
            SJ[g,:] = logpdf_GAU_ND(X, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        gmmnew = GMM
        P = numpy.exp(SJ - SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (toRow(gamma)*X).sum(1)
            S = numpy.dot(X, (toRow(gamma)*X).T)
            w = Z/N
            mu = toCol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            # diag sigma
            if model == "diagnal":
                Sigma = Sigma * numpy.eye(Sigma.shape[0])
            # to apply threshold for EValues
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < thresholdForEValues ] = thresholdForEValues
            SigmaNew = numpy.dot(U, toCol(s) * U.T)
            gmmNew.append((w,mu,SigmaNew))
        if model == "tied":
            # print("tied")
            sigTot = sum([w * sig for w, mu, sig in gmmNew])
            gmmNew = [(w,mu, sigTot) for w, mu, s in gmmNew]

        GMM = gmmNew
        # print(llNew)
    # print(llNew - llOld)
    return gmmnew

def GMM_SJoint(X, GMM):
    G = len(GMM)
    N = X.shape[1]
    SJ = numpy.zeros((G, N))
    for g in range(G):
        SJ[g, :] = logpdf_GAU_ND(X, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
    return SJ

def GMM_ll_PerSample(X,GMM):
    G = len(GMM)
    N= X.shape[1]
    S= numpy.zeros((G,N))
    for g in range(G):
        S[g,:] = logpdf_GAU_ND(X, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
    return scipy.special.logsumexp(S, axis = 0)

def gmmlbg(GMM,alpha):
    G = len(GMM)
    newGMM = []
    for g in range(G):
        (w, mu, CovarianMatrix) = GMM[g]
        U, s, _ = numpy.linalg.svd(CovarianMatrix)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        newGMM.append((w / 2, mu - d, CovarianMatrix ))
        newGMM.append((w / 2, mu + d, CovarianMatrix ))
    return newGMM


class GMM:
    def __init__(self, kfold, thresholdForEValues=0.1, numberOfComponents=2, model="full", pca=False, zNorm=False, pcaM=12,tqdm=True, eval=False):
        self.dir = 'result/gmm/'+model+"/"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.dir_pca = 'result/gmm/pca/'+model+"/"
        if not os.path.exists(self.dir_pca ):
            os.makedirs(self.dir_pca)

        self.dir_eval = 'result/gmm/eval/' + model + "/"
        if not os.path.exists(self.dir_eval):
            os.makedirs(self.dir_eval)

        self.allFolds = kfold
        self.thresholdForEValues = thresholdForEValues
        self.numberOfComponents = numberOfComponents
        self.model = model
        self.tqdm = not tqdm

        self.pca = pca
        self.zNorm = zNorm
        self.pcaM = pcaM

        self.eval=eval

        self.groundTruthLabels = list()
        self.scores = list()
        self.prediction = list()
        self.train()


    def train(self):
        for (Xtrain, Ytrain), (Xtest, Ytest) in tqdm(self.allFolds, disable=self.tqdm):
            Xtrain, Xtest = preprocessing(Xtrain=Xtrain, Xtest=Xtest, zNorm=self.zNorm, pca=self.pca, pcaM=self.pcaM)
            mu = mean(Xtrain)
            C = covarinceMartix(Xtrain)
            gmm_init_0 = [(1.0, mu, C)]
            NewgmmEM = 0

            itteration = 1 + int(numpy.log2(self.numberOfComponents))

            mu_Cov_weight_pair_each_class = {}

            for label in set(list(Ytrain)):
                # print("label=", label)
                gmm_init = gmm_init_0
                for i in range(itteration):
                    # print("GMM LEN", len(gmm_init))
                    NewgmmEM = GMM_EM(Xtrain[:, Ytrain == label], gmm_init, self.thresholdForEValues, self.model)
                    if i < itteration - 1:
                        gmm_init = gmmlbg(NewgmmEM, alpha=0.1)
                mu_Cov_weight_pair_each_class[label] = NewgmmEM

            final = numpy.zeros((len(set(Ytest)), Xtest.shape[1]))
            for i in set(list(Ytrain)):
                GMM = mu_Cov_weight_pair_each_class[i]
                SM = GMM_ll_PerSample(Xtest, GMM)
                final[i] = SM

            llr = final[1, :] - final[0, :]

            predictedLabelByGMM = final.argmax(0)
            self.groundTruthLabels.append(Ytest)
            self.scores.append(llr)
            self.prediction.append(predictedLabelByGMM)
            # return predictedLabelByGMM, llr

        self.logLiklihoodRatio = numpy.hstack(self.scores)
        self.groundTruthLabels = numpy.hstack(self.groundTruthLabels)
        self.predictions = numpy.hstack(self.prediction)
        fileName = "gmm_components_" + str(self.numberOfComponents) +"_pca_" + str(self.pca) + "_" + str(self.pcaM) + "_" + "zNorm_" + str(self.zNorm)
        labelFileName = "label_gmm_components_" + str(self.numberOfComponents) +"_pca_" + str(self.pca) + "_" + str(self.pcaM) + "_" + "zNorm_" + str(self.zNorm)
        if not self.eval:
            if not self.pca:
                with open(self.dir + fileName, 'wb') as f:
                    pickle.dump(self.logLiklihoodRatio, f)
                with open(self.dir + labelFileName, 'wb') as f:
                    pickle.dump(self.groundTruthLabels, f)
            else:
                with open(self.dir_pca + fileName, 'wb') as f:
                    pickle.dump(self.logLiklihoodRatio, f)
                with open(self.dir_pca + labelFileName, 'wb') as f:
                    pickle.dump(self.groundTruthLabels, f)
        else:
            with open(self.dir_eval + fileName, 'wb') as f:
                pickle.dump(self.logLiklihoodRatio, f)
            with open(self.dir_eval + labelFileName, 'wb') as f:
                pickle.dump(self.groundTruthLabels, f)

