import numpy
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
from matplotlib import colors

import load
from util import *
from tqdm import tqdm
import os

def heatMap(correlationMatrix, classs=2):
    '''
        'viridis': A perceptually uniform colormap that is often used for sequential data.
        'cool': A colormap with cool tones, ranging from cyan to magenta.
        'hot': A colormap with hot tones, ranging from black to red and white.
        'bwr': A colormap that emphasizes both positive and negative values, with blue for
    '''

    directory = 'plots/heat maps/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plot.subplots()

    if classs == 2:
        im = ax.imshow(correlationMatrix, cmap='viridis')
        ax.set_title("Pearson's Correlation Coefficient Heatmap for Both Classes")

    elif classs == 0:
        im = ax.imshow(correlationMatrix, cmap='cool')
        ax.set_title("Pearson's Correlation Coefficient Heatmap for Class 0")

    else:
        im = ax.imshow(correlationMatrix, cmap='hot')
        ax.set_title("Pearson's Correlation Coefficient Heatmap for Class 1")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    ax.set_xlabel("Features")
    ax.set_ylabel("Features")
    if classs == 2:
        plot.savefig(directory+'heat map data set.png')
    elif classs == 1:
        plot.savefig(directory+'heat map class_%d.png' % classs)
    else:
        plot.savefig(directory + 'heat map class_%d.png' % classs)
    plot.show()


def scatter(samples, labels):
    class0Mask = labels == 0
    class1Mask = labels == 1

    samplesClass0 = samples[:, class0Mask]
    samplesClass1 = samples[:, class1Mask]

    directory = 'plots/scatter/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(samples.shape[0]):
        for j in range(samples.shape[0]):
            if i == j:
                continue
            else:
                plot.scatter(samplesClass0[i, :], samplesClass0[j, :], label="Male", alpha=0.2,  s=80)
                plot.scatter(samplesClass1[i, :], samplesClass1[j, :], label="Female", alpha=0.2,  s=80)
                plot.xlabel("feature_%d" % j)
                plot.ylabel("feature_%d" % i)
                plot.legend()
                plot.savefig('plots/scatter/scatter_feature_%d_%d.png' % (i, j))
                plot.show()

def histogram(samples, labels):
    class0Mask = labels == 0
    class1Mask = labels == 1

    directory = 'plots/histogram'
    if not os.path.exists(directory):
        os.makedirs(directory)


    samplesClass0 = samples[:, class0Mask]
    samplesClass1 = samples[:, class1Mask]

    for feature in range(samples.shape[0]):
        plot.hist(samplesClass0[feature,:], density=True, bins=50, edgecolor='k', alpha = 0.4, label="Male")
        plot.hist(samplesClass1[feature,:], density=True, bins=50, edgecolor='k', alpha = 0.4,label="Female")
        plot.xlabel("feature_%d" % feature)
        plot.legend()
        plot.savefig('plots/histogram/hist_feature_%d.png' % feature)
        plot.show()

def roc_curve_of_best_models(scores, groundTruthLabels, workingPoint, model=None):

    for j in range(2):
        llr = scores[j].ravel()
        thresholds = numpy.array(llr.ravel())
        thresholds.sort()
        thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
        groundTruthLabelsw = groundTruthLabels[j].ravel()
        tpr = numpy.zeros(thresholds.size)
        fpr = numpy.zeros(thresholds.size)
        for idx, t in enumerate(thresholds):
            predic = numpy.int32(llr > t)
            mp = MeasuringPredictions(ground_truth_labels=groundTruthLabelsw, predicted_labels=predic, target_prior=workingPoint.piTilde, fpC=1.0, fnC=1.0)
            tpr[idx] = mp.tpR
            fpr[idx] = mp.fpR
        plot.plot(fpr, tpr, label=model[j])

    plot.legend()
    plot.ylabel("TPR")
    plot.xlabel("FPR")
    plot.grid(True)
    plot.show()

def roc_curve(scores, groundTruthLabels, workingPoint, model=None):
    llr = scores.ravel()
    thresholds = numpy.array(llr.ravel())
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    groundTruthLabels = groundTruthLabels.ravel()
    tpr = numpy.zeros(thresholds.size)
    fpr = numpy.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        predic = numpy.int32(llr > t)
        mp = MeasuringPredictions(ground_truth_labels=groundTruthLabels, predicted_labels=predic,
                                  target_prior=workingPoint.piTilde, fpC=1.0, fnC=1.0)
        tpr[idx] = mp.tpR
        fpr[idx] = mp.fpR
    if model == "G":
        model = "multivariate gaussian"
    elif model == "N":
        model = "naive bayes gaussian"
    elif model == "T":
        model = "tied gaussian"

    plot.plot(fpr, tpr, label=model)
    plot.legend()
    plot.grid(True)
    plot.show()


def error_plot_lambda_rang_lr(allFolds, workingPoint, pca=False, zNorm=False, pcaM=12, title=None, save=False):
    directory = 'plots/lr plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directoryResult = "result/lr/error_plot_lambda_rang_lr/"
    if not os.path.exists(directoryResult):
        os.makedirs(directoryResult)

    lowerBound = -4
    upperBound = 2
    lambdas = 15
    color = ["r", "g", "b"]
    rangeOfLambda = 10**numpy.linspace(lowerBound, upperBound, lambdas)
    DCF_for_priors = {}
    # DCF_for_priors2 = {}
    if not save:
        piTildes = [0.5, 0.1, 0.9]
    else:
        piTildes = [0.5]

    for piTilde in piTildes:
        print(f"piTilde={piTilde}")
        plot_min_dcf = []

        for i in tqdm(rangeOfLambda):
            workingPointForMinDCF = WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0)
            lr = LogisticRegression(kfold=allFolds, workingPoint=workingPoint, lambdaa=i, pca=pca, zNorm=zNorm, pcaM=pcaM, tqdm=False, save=save)
            minimumDCF, _ = min_DCF(lr.logLiklihoodRatio, lr.groundTruthLabels, workingPoint=workingPointForMinDCF)
            plot_min_dcf.append(minimumDCF)

        DCF_for_priors[piTilde] = [plot_min_dcf, workingPointForMinDCF.piTilde]

        if not save:
            fileName = "lr_gamma_" + "range" + "_pca_" + str(pca) + "_" + str(
                pcaM) + "_" + "zNorm_" + str(zNorm) + "_piThilde_" + str(workingPointForMinDCF.piTilde)
            with open(directoryResult + fileName, 'wb') as f:
                pickle.dump(plot_min_dcf, f)
    # data = load.Load("Data/train.txt")
    # dataTest = load.Load("Data/test.txt")
    # for piTilde in piTildes:
    #     print(f"piTilde={piTilde}")
    #     plot_min_dcf = []
    #
    #     for i in tqdm(rangeOfLambda):
    #         workingPointForMinDCF = WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0)
    #         lr = LogisticRegression(kfold=[((data.samples, data.labels),(dataTest.samples, dataTest.labels))], workingPoint=workingPoint, lambdaa=i, pca=pca, zNorm=zNorm,
    #                                 pcaM=pcaM, tqdm=False, save=save)
    #         minimumDCF, _ = min_DCF(lr.logLiklihoodRatio, lr.groundTruthLabels, workingPoint=workingPointForMinDCF)
    #         plot_min_dcf.append(minimumDCF)
    #
    #     DCF_for_priors2[piTilde] = [plot_min_dcf, workingPointForMinDCF.piTilde]


    plot.xlabel("\u03BB")
    plot.xscale("log")
    plot.ylabel("minDCF")
    for index, piTilde in enumerate(piTildes):
        plot.plot(rangeOfLambda, DCF_for_priors[piTilde][0], color=color[index], label='minDCF(\u03C0\u0303̂=' + str(DCF_for_priors[piTilde][1])+')[Val]')

    # for index, piTilde in enumerate(piTildes):
    #     plot.plot(rangeOfLambda, DCF_for_priors2[piTilde][0], color=color[index], linestyle='--', label='minDCF(\u03C0\u0303̂=' + str(DCF_for_priors2[piTilde][1]) + ')[Eval]')

    plot.legend()
    if title is not None:
        plot.title(title)
    plot.grid(True)
    # plot.savefig(directory + title + '.png')
    plot.show()

def error_plot_c_range_linear_and_poly_svm(allFolds, recalanceClasses = False, title=None, rbf=False ,poly=False, pca=False, zNorm=False, pcaM=12):
    directory = 'plots/svm plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    lowerBound = -5
    upperBound = 2
    Cs = 15
    color = ["r", "g", "b"]
    rangeOfCs = 10**numpy.linspace(lowerBound, upperBound, Cs)
    # rangeOfCs = numpy.exp(numpy.linspace(lowerBound, upperBound, Cs))
    lambdaa = 1.e-05
    DCF_for_priors = {}
    DCF_for_priors2 = {}
    piTildes = [0.5, 0.1, 0.9]

    for piTilde in piTildes:
        print(f"piTilde={piTilde}")
        plot_min_dcf = []
        for i in tqdm(rangeOfCs):
            workingPoint = WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0)
            if recalanceClasses:
                svm = SupportVectorMachine(kfold=allFolds, rbf=rbf, polynomial=poly, C=i, targetPrior=piTilde ,gamma=1.0, pca=pca, pcaM=pcaM, zNorm=zNorm, tqdm=False)
            else:
                svm = SupportVectorMachine(kfold=allFolds, rbf=rbf, polynomial=poly, C=i, targetPrior=None, gamma=1.0, pca=pca, pcaM=pcaM, zNorm=zNorm, tqdm=False)
            # kfoldCalibration = KFold(toRow(svm.allScores), svm.groundTrouthLabels, n_splits=3)
            # lr_svm = LogisticRegression(kfold=kfoldCalibration.allFolds, workingPoint=workingPoint, lambdaa=lambdaa, pca=False,
            #                             zNorm=False, tqdm=False)
            # minimumDCF, _ = min_DCF(lr_svm.logLiklihoodRatio, lr_svm.groundTruthLabels, workingPoint=workingPoint)
            minimumDCF, _ = min_DCF(svm.allScores, svm.groundTrouthLabels, workingPoint=workingPoint)
            plot_min_dcf.append(minimumDCF)
        DCF_for_priors[piTilde] = [plot_min_dcf, workingPoint.piTilde]

    data = load.Load("Data/train.txt")
    dataTest = load.Load("Data/test.txt")


    for piTilde in piTildes:
        print(f"piTilde={piTilde}")
        plot_min_dcf = []
        for i in tqdm(rangeOfCs):
            workingPoint = WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0)
            if recalanceClasses:
                svm = SupportVectorMachine(kfold=[((data.samples, data.labels), (dataTest.samples, dataTest.labels))], rbf=rbf, polynomial=poly, C=i, targetPrior=piTilde ,gamma=1.0, pca=pca, pcaM=pcaM, zNorm=zNorm, tqdm=False)
            else:
                svm = SupportVectorMachine(kfold=[((data.samples, data.labels), (dataTest.samples, dataTest.labels))], rbf=rbf, polynomial=poly, C=i, targetPrior=None, gamma=1.0, pca=pca, pcaM=pcaM, zNorm=zNorm, tqdm=False)
            minimumDCF, _ = min_DCF(svm.allScores, svm.groundTrouthLabels, workingPoint=workingPoint)
            plot_min_dcf.append(minimumDCF)
        DCF_for_priors2[piTilde] = [plot_min_dcf, workingPoint.piTilde]


    plot.xlabel("C")
    plot.xscale("log")
    plot.ylabel("minDCF")
    for index, piTilde in enumerate(piTildes):
        plot.plot(rangeOfCs, DCF_for_priors[piTilde][0], color=color[index], label='minDCF(\u03C0\u0303̂=' + str(DCF_for_priors[piTilde][1])+')[Val]')

    for index, piTilde in enumerate(piTildes):
        plot.plot(rangeOfCs, DCF_for_priors2[piTilde][0], color=color[index], linestyle='--', label='minDCF(\u03C0\u0303̂=' + str(DCF_for_priors2[piTilde][1])+')[Eval]')

    if title is not None:
        plot.title(title)
    plot.legend()
    plot.yticks(numpy.linspace(0, 1.1, 12))
    plot.grid(True)
    plot.savefig(directory + title + '.png')
    plot.show()
# do not forget to pass working point to the function, in this version it's hard coded with wp in error_plot_c_range_linear_and_poly_svm function. the same for other wrappers
def error_plot_c_range_linear_and_poly_svm_wrapper(allFolds, trget=None, rbf=False, poly=False, pca=False, zNorm=False, pcaM=12):
    if not poly and not rbf:
        if trget is None:
            error_plot_c_range_linear_and_poly_svm(allFolds=allFolds, recalanceClasses=False,
                                                        title="Evaluation - SVM-Linear(normal)", rbf=rbf, poly=poly, pca=pca, zNorm=zNorm, pcaM=pcaM)
        else:
            error_plot_c_range_linear_and_poly_svm(allFolds=allFolds, recalanceClasses=True,
                                                        title="Evaluation - SVM-Linear(class balanced)", rbf=rbf, poly=poly, pca=pca, zNorm=zNorm, pcaM=pcaM)
    elif poly:
        if trget is None:
            error_plot_c_range_linear_and_poly_svm(allFolds=allFolds, recalanceClasses=False,
                                                        title="SVM-Polynomial(normal)", rbf=rbf, poly=poly, pca=pca, zNorm=zNorm, pcaM=pcaM)
        else:
            error_plot_c_range_linear_and_poly_svm(allFolds=allFolds, recalanceClasses=True,
                                                        title="SVM-Polynomial(class balanced)", rbf=rbf, poly=poly, pca=pca, zNorm=zNorm, pcaM=pcaM)
    elif rbf:
        print("rbff")
        if trget is None:
            error_plot_c_range_rbf_svm_with_different_gammas(allFolds=allFolds, recalanceClasses=False,
                                                        title="SVM-RBF(normal)", pca=pca, zNorm=zNorm, pcaM=pcaM)
        else:
            error_plot_c_range_rbf_svm_with_different_gammas(allFolds=allFolds, recalanceClasses=True,
                                                        title="SVM-RBF(class balanced)", pca=pca, zNorm=zNorm, pcaM=pcaM)




def error_plot_c_range_rbf_svm_with_different_gammas(allFolds, recalanceClasses = False, title=None, pca=False, zNorm=False, pcaM=12):
    directory = 'plots/svm plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    lowerBound = -3
    upperBound = 3
    Cs = 15
    color = ["r", "g", "b"]
    rangeOfCs = 10**numpy.linspace(lowerBound, upperBound, Cs)
    lambdaa = 1.e-05
    DCF_for_gammas = {}
    DCF_for_gammas2 = {}
    gammas = [-3, -4, -5]

    for gammaa in gammas:
        print(f"gamma={numpy.exp(gammaa)}")
        plot_min_dcf = []
        for i in tqdm(rangeOfCs):
            workingPoint = WorkingPoint(target_prior=0.5, fnC=1.0, fpC=1.0)
            if recalanceClasses:
                svm = SupportVectorMachine(kfold=allFolds, rbf=True, polynomial=False, C=i, targetPrior=workingPoint.piTilde ,gamma=numpy.exp(gammaa), pca=pca, zNorm=zNorm, pcaM=pcaM, tqdm=False)
            else:
                svm = SupportVectorMachine(kfold=allFolds, rbf=True, polynomial=False, C=i, targetPrior=None, gamma=numpy.exp(gammaa), pca=pca, zNorm=zNorm, pcaM=pcaM, tqdm=False)
            # kfoldCalibration = KFold(toRow(svm.allScores), svm.groundTrouthLabels, n_splits=3, seed=40)
            # lr_svm = LogisticRegression(kfold=kfoldCalibration.allFolds, workingPoint=workingPoint, lambdaa=lambdaa, pca=False, zNorm=False, tqdm=False)
            minimumDCF, _ = min_DCF(svm.allScores, svm.groundTrouthLabels, workingPoint=workingPoint)
            plot_min_dcf.append(minimumDCF)
        DCF_for_gammas[gammaa] = [plot_min_dcf, gammaa]



    # data = load.Load("Data/train.txt")
    # dataTest = load.Load("Data/test.txt")
    #
    # for gammaa in gammas:
    #     print(f"gamma={numpy.exp(gammaa)}")
    #     plot_min_dcf = []
    #     for i in tqdm(rangeOfCs):
    #         workingPoint = WorkingPoint(target_prior=0.5, fnC=1.0, fpC=1.0)
    #         if recalanceClasses:
    #             svm = SupportVectorMachine(kfold=[((data.samples, data.labels),(dataTest.samples, dataTest.labels))], rbf=True, polynomial=False, C=i,
    #                                        targetPrior=workingPoint.piTilde, gamma=numpy.exp(gammaa), pca=pca,
    #                                        zNorm=zNorm, pcaM=pcaM, tqdm=False)
    #         else:
    #             svm = SupportVectorMachine(kfold=[((data.samples, data.labels),(dataTest.samples, dataTest.labels))], rbf=True, polynomial=False, C=i, targetPrior=None,
    #                                        gamma=numpy.exp(gammaa), pca=pca, zNorm=zNorm, pcaM=pcaM, tqdm=False)
    #         # kfoldCalibration = KFold(toRow(svm.allScores), svm.groundTrouthLabels, n_splits=3, seed=40)
    #         # lr_svm = LogisticRegression(kfold=kfoldCalibration.allFolds, workingPoint=workingPoint, lambdaa=lambdaa, pca=False, zNorm=False, tqdm=False)
    #         minimumDCF, _ = min_DCF(svm.allScores, svm.groundTrouthLabels, workingPoint=workingPoint)
    #         plot_min_dcf.append(minimumDCF)
    #     DCF_for_gammas2[gammaa] = [plot_min_dcf, gammaa]





    plot.xlabel("C")
    plot.xscale("log")
    plot.ylabel("minDCF")
    ymax = list()
    for index, gamma in enumerate(gammas):
        ymax.append(max(DCF_for_gammas[gamma][0]))
        plot.plot(rangeOfCs, DCF_for_gammas[gamma][0], color=color[index], label='[Val]log \u03B3=' + str(DCF_for_gammas[gamma][1]))

    # for index, gamma in enumerate(gammas):
    #     ymax.append(max(DCF_for_gammas2[gamma][0]))
    #     plot.plot(rangeOfCs, DCF_for_gammas2[gamma][0], color=color[index], linestyle='--', label='[Eval]log \u03B3=' + str(DCF_for_gammas2[gamma][1]))

    if title is not None:
        plot.title(title)
    plot.legend()
    ymax = max(ymax) + 0.1
    plot.ylim(0, ymax)
    # plot.yticks(numpy.linspace(0, 1.1, 12))
    plot.grid(True)
    # plot.savefig(directory+'grid search c and gamma rbf'+title+'.png')
    plot.show()



def bayesErrorPlot(scores, groundTruthLabels, title = None, model=None):
    directory = 'plots/bayes error plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    color=["r", "g", "b"]
    low = -3
    high = 3
    division = 15
    effPriorLogOdds = numpy.linspace(low, high, division)
    for j in range(2):
        normalizeDCFs = []
        min_normalizeDCFs = []
        for i in tqdm(effPriorLogOdds):
            wp = WorkingPoint(target_prior=i, fnC=1.0, fpC=1.0, errorPlot=True)
            prediction = (scores[j] > wp.effectiveThresshold)* 1
            mp = MeasuringPredictions(ground_truth_labels=groundTruthLabels[j], predicted_labels=prediction, fnC=1.0, fpC=1.0, target_prior=wp.piTilde)
            normalizeDCF = mp.normalDCF
            normalizeDCFs.append(normalizeDCF)
            min_normalizeDCF,  _ = min_DCF(llr=scores[j], groundTruthLabels=groundTruthLabels[j], workingPoint=wp)
            min_normalizeDCFs.append(min_normalizeDCF)
        plot.plot(effPriorLogOdds, numpy.array(normalizeDCFs), label=model[j]+'- actualDCF', color=color[j])
        plot.plot(effPriorLogOdds, numpy.array(min_normalizeDCFs), linestyle='--', label=model[j]+'- minDCF', color=color[j])


    plot.xlabel(r'$log\frac{\widetilde{\pi}}{1-\widetilde{\pi}}$')
    plot.ylabel("DCF")
    if title is not None:
        plot.title(title)
    # plot.yticks(numpy.linspace(0, 1, 11))
    # ymax = max(normalizeDCFs) + 0.1
    # plot.ylim(0, ymax)
    plot.xlim([low, high])
    plot.legend()
    plot.grid(True)
    plot.savefig(directory + title + '.png')
    plot.show()


def error_bayes_plot_wrapper_svm(scores, groundTruthLabels, rbf=False, poly=False, trget=None):
    if not rbf and not poly:
        if trget is None:
            bayesErrorPlot(scores=scores, groundTruthLabels=groundTruthLabels,
                                title="Bayes Error Plot For SVM-linear(calibrated scores-normal)")
        else:
            bayesErrorPlot(scores=scores, groundTruthLabels=groundTruthLabels,
                                title="Bayes Error Plot For SVM-linear(calibrated scores-class balanced)")
    elif rbf:
        if trget is None:
            bayesErrorPlot(scores=scores, groundTruthLabels=groundTruthLabels,
                                title="Bayes Error Plot For SVM-RBF(calibrated scores-normal)")
        else:
            bayesErrorPlot(scores=scores, groundTruthLabels=groundTruthLabels,
                                title="Bayes Error Plot For SVM-RBF(calibrated scores-class balanced)")
    else:
        if trget is None:
            bayesErrorPlot(scores=scores, groundTruthLabels=groundTruthLabels,
                                title="Bayes Error Plot For SVM-Polynomial(calibrated scores-class normal)")
        else:
            bayesErrorPlot(scores=scores, groundTruthLabels=groundTruthLabels,
                                title="Bayes Error Plot For SVM-Polynomial(calibrated scores-class balanced)")

def error_plot_gmm_different_components(allFolds, piTilde = 0.5, rangeComponents=[1, 2, 4, 8], title=None, calibration=False, pca=False, gaussianization=False, pcaM=12):
    directory = 'plots/gmm plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    DCF_for_priors = {}
    lambdaa = 1.e-5

    for component in tqdm(rangeComponents):
        plot_min_dcf = []
        gmm = GMM(kfold=allFolds, numberOfComponents=component, tqdm=False)
        if calibration:
            kfoldCalibration = KFold(toRow(gmm.logLiklihoodRatio), gmm.groundTruthLabels, n_splits=3, seed=70)
            lr_gmm = LogisticRegression(kfold=kfoldCalibration.allFolds, workingPoint=WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0), lambdaa=lambdaa, pca=False,
                                        gaussianization=False, tqdm=False)
            minimumDCF, _ = min_DCF(lr_gmm.logLiklihoodRatio, lr_gmm.groundTruthLabels, workingPoint=WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0))
            plot_min_dcf.append(minimumDCF)
            DCF_for_priors[component] = [minimumDCF, piTilde]
        else:
            minimumDCF, _ = min_DCF(gmm.logLiklihoodRatio, gmm.groundTruthLabels, workingPoint=WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0))
            # plot_min_dcf.append(minimumDCF)
            DCF_for_priors[component] = [minimumDCF, piTilde]
            # DCF_for_priors[component] = [plot_min_dcf, piTilde]


    plot.xlabel("GMM Components")
    plot.ylabel("minDCF")
    bar_width = 0.05
    for index, component in enumerate(rangeComponents):
        plot.bar(index+1 - bar_width - 0.02 / 2, DCF_for_priors[component][0] ,bar_width, color="r", label='1')

    for index, component in enumerate(rangeComponents):
        plot.bar(index+1 + bar_width + 0.02 / 2, DCF_for_priors[component][0]+0.18, bar_width, color="g",label='2')

    if title is not None:
        plot.title(title)
    # plot.legend(['minDCF(\u03C0\u0303̂=0.5)', 'minDCF1(\u03C0\u0303̂=0.5)'])

    line1_patch = mpatches.Patch(color='r', label='Line 1')
    line2_patch = mpatches.Patch(color='g', label='Line 2')
    plot.legend(handles=[line1_patch, line2_patch])

    # plot.legend(['1', '2'])
    temp = list(range(len(rangeComponents)))[1:]
    temp.append(len(rangeComponents))
    print(temp)
    plot.xticks(temp, rangeComponents)
    plot.grid(True)
    plot.savefig(directory + title + '.png')
    plot.show()

def error_plot_gmm_different_components(piTilde=0.5, title=None, model="full model"):
    directory = 'plots/gmm plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    rangeComponents = [1, 2, 4, 8, 16]

    DCF_without_PCA = {}
    DCF_with_PCA_12 = {}
    DCF_with_PCA_11 = {}
    DCF_with_PCA_10 = {}

    # DCF_for_priors[component] = [minimumDCF, piTilde]
    if model == "full model":
        data_without_PCA= ["full/gmm_components_1_pca_False_11_zNorm_False",
                           "full/gmm_components_2_pca_False_11_zNorm_False",
                           "full/gmm_components_4_pca_False_11_zNorm_False",
                           "full/gmm_components_8_pca_False_11_zNorm_False",
                           "full/gmm_components_16_pca_False_11_zNorm_False"]
        label_without_PCA= ["full/label_gmm_components_1_pca_False_11_zNorm_False",
                            "full/label_gmm_components_2_pca_False_11_zNorm_False",
                            "full/label_gmm_components_4_pca_False_11_zNorm_False",
                            "full/label_gmm_components_8_pca_False_11_zNorm_False",
                            "full/label_gmm_components_16_pca_False_11_zNorm_False"]
    else:
        data_without_PCA = ["tied/gmm_components_1_pca_False_11_zNorm_False",
                            "tied/gmm_components_2_pca_False_11_zNorm_False",
                            "tied/gmm_components_4_pca_False_11_zNorm_False",
                            "tied/gmm_components_8_pca_False_11_zNorm_False",
                            "tied/gmm_components_16_pca_False_11_zNorm_False"]
        label_without_PCA = ["tied/label_gmm_components_1_pca_False_11_zNorm_False",
                             "tied/label_gmm_components_2_pca_False_11_zNorm_False",
                             "tied/label_gmm_components_4_pca_False_11_zNorm_False",
                             "tied/label_gmm_components_8_pca_False_11_zNorm_False",
                             "tied/label_gmm_components_16_pca_False_11_zNorm_False"]


    # data_with_PCA_12 = ["gmm_components_1_pca_True_12_zNorm_False",
    #                     "gmm_components_2_pca_True_12_zNorm_False",
    #                     "gmm_components_4_pca_True_12_zNorm_False",
    #                     "gmm_components_8_pca_True_12_zNorm_False",
    #                     "gmm_components_16_pca_True_12_zNorm_False",]
    # label_with_PCA_12 = ["label_gmm_components_1_pca_True_12_zNorm_False",
    #                      "label_gmm_components_2_pca_True_12_zNorm_False",
    #                      "label_gmm_components_4_pca_True_12_zNorm_False",
    #                      "label_gmm_components_8_pca_True_12_zNorm_False",
    #                      "label_gmm_components_16_pca_True_12_zNorm_False",]
    #
    # data_with_PCA_11 = ["gmm_components_1_pca_True_11_zNorm_False",
    #                     "gmm_components_2_pca_True_11_zNorm_False",
    #                     "gmm_components_4_pca_True_11_zNorm_False",
    #                     "gmm_components_8_pca_True_11_zNorm_False",
    #                     "gmm_components_16_pca_True_11_zNorm_False",]
    # label_with_PCA_11 = ["label_gmm_components_1_pca_True_11_zNorm_False",
    #                      "label_gmm_components_2_pca_True_11_zNorm_False",
    #                      "label_gmm_components_4_pca_True_11_zNorm_False",
    #                      "label_gmm_components_8_pca_True_11_zNorm_False",
    #                      "label_gmm_components_16_pca_True_11_zNorm_False",]
    #
    # data_with_PCA_10 = ["gmm_components_1_pca_True_10_zNorm_False",
    #                     "gmm_components_2_pca_True_10_zNorm_False",
    #                     "gmm_components_4_pca_True_10_zNorm_False",
    #                     "gmm_components_8_pca_True_10_zNorm_False",
    #                     "gmm_components_16_pca_True_10_zNorm_False", ]
    # label_with_PCA_10 = ["label_gmm_components_1_pca_True_10_zNorm_False",
    #                      "label_gmm_components_2_pca_True_10_zNorm_False",
    #                      "label_gmm_components_4_pca_True_10_zNorm_False",
    #                      "label_gmm_components_8_pca_True_10_zNorm_False",
    #                      "label_gmm_components_16_pca_True_10_zNorm_False", ]

    for index in range(len(data_without_PCA)):
        adr_data = data_without_PCA[index]
        adr_lbl = label_without_PCA[index]

        with open(os.getcwd()+'/result/gmm/eval/'+adr_data, 'rb') as f:
            data = pickle.load(f)

        with open(os.getcwd()+'/result/gmm/eval/'+adr_lbl, 'rb') as f:
            label = pickle.load(f)

        minimumDCF, _ = min_DCF(data, label, workingPoint=WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0))
        DCF_without_PCA[index] = [minimumDCF, piTilde]

    # for index in range(len(data_with_PCA_12)):
    #     adr_data = data_with_PCA_12[index]
    #     adr_lbl = label_with_PCA_12[index]
    #
    #     with open(os.getcwd() + '/result/gmm/pca/' + adr_data, 'rb') as f:
    #         data = pickle.load(f)
    #
    #     with open(os.getcwd() + '/result/gmm/pca/' + adr_lbl, 'rb') as f:
    #         label = pickle.load(f)
    #
    #
    #     minimumDCF, _ = min_DCF(data, label, workingPoint=WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0))
    #     DCF_with_PCA_12[index] = [minimumDCF, piTilde]
    #
    # for index in range(len(data_with_PCA_11)):
    #     adr_data = data_with_PCA_11[index]
    #     adr_lbl = label_with_PCA_11[index]
    #
    #     with open(os.getcwd() + '/result/gmm/pca/' + adr_data, 'rb') as f:
    #         data = pickle.load(f)
    #
    #     with open(os.getcwd() + '/result/gmm/pca/' + adr_lbl, 'rb') as f:
    #         label = pickle.load(f)
    #
    #
    #     minimumDCF, _ = min_DCF(data, label, workingPoint=WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0))
    #     DCF_with_PCA_11[index] = [minimumDCF, piTilde]
    #
    # for index in range(len(data_with_PCA_10)):
    #     adr_data = data_with_PCA_10[index]
    #     adr_lbl = label_with_PCA_10[index]
    #
    #     with open(os.getcwd() + '/result/gmm/pca/' + adr_data, 'rb') as f:
    #         data = pickle.load(f)
    #
    #     with open(os.getcwd() + '/result/gmm/pca/' + adr_lbl, 'rb') as f:
    #         label = pickle.load(f)


        # minimumDCF, _ = min_DCF(data, label, workingPoint=WorkingPoint(target_prior=piTilde, fnC=1.0, fpC=1.0))
        # DCF_with_PCA_10[index] = [minimumDCF, piTilde]


    plot.xlabel("GMM Components")
    plot.ylabel("DCF")
    bar_width = 0.05

    for index, component in enumerate(rangeComponents):
        plot.bar(index+1, DCF_without_PCA[index][0] ,bar_width, color="r", label='Raw features')

    # for index, component in enumerate(rangeComponents):
    #     plot.bar(index+1 - 2*bar_width - 2*0.02 / 2, DCF_without_PCA[index][0] ,bar_width, color="r", label='Raw features')

    # for index, component in enumerate(rangeComponents):
    #     plot.bar(index+1 - bar_width - 0.02 / 2, DCF_with_PCA_12[index][0], bar_width, color="g", label='minDCF(\u03C0\u0303=0.5) - PCA(m=12)')
    #
    # for index, component in enumerate(rangeComponents):
    #     plot.bar(index+1 + bar_width + 0.02 / 2, DCF_with_PCA_11[index][0], bar_width, color="b", label='minDCF(\u03C0\u0303=0.5) - PCA(m=11)')
    #
    # for index, component in enumerate(rangeComponents):
    #     plot.bar(index + 1 +  2 * bar_width + 2 * 0.02 / 2, DCF_with_PCA_10[index][0], bar_width, color="y", label='minDCF(\u03C0\u0303=0.5) - PCA(m=10)')

    if title is not None:
        plot.title(title + " - " + model)
    # plot.legend(['minDCF(\u03C0\u0303̂=0.5)', 'minDCF1(\u03C0\u0303̂=0.5)'])

    line1_patch = mpatches.Patch(color='r', label='Raw features')
    # line2_patch = mpatches.Patch(color='g', label='minDCF(\u03C0\u0303=0.5) - PCA(m=12)')
    # line3_patch = mpatches.Patch(color='b', label='minDCF(\u03C0\u0303=0.5) - PCA(m=11)')
    # line4_patch = mpatches.Patch(color='y', label='minDCF(\u03C0\u0303=0.5) - PCA(m=10)')
    # plot.legend(handles=[line1_patch, line2_patch, line3_patch, line4_patch])
    plot.legend(handles=[line1_patch])


    temp = list(range(len(rangeComponents)))[1:]
    temp.append(len(rangeComponents))
    plot.yticks(numpy.linspace(0, 0.225, 5))
    plot.xticks(temp, rangeComponents)
    plot.grid(True)
    plot.savefig(directory + title + model + '.png')
    plot.show()
