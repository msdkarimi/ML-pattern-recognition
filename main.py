import numpy
import load
import os
import plot
import util
from util import KFold
from util import KFold
from util import MVG
from util import WorkingPoint
from util import *

import matplotlib.pyplot as plt

def main():

    # load data
    data = load.Load("Data/train.txt")
    dataTest = load.Load("Data/test.txt")

    # kfold
    kfold = KFold(data.samples, data.labels, n_splits=3)
    # define working point
    wp = WorkingPoint(target_prior=0.5, fnC=1.0, fpC=1.0)

#     -------------------------------------------------------------------- PLOTS ------------------------------------------------------------------

    # pearsonsCorrelationCoefficient = computeCorrelationMatrix(data.samples)
    # pearsonsCorrelationCoefficient = computeCorrelationMatrix(data.samples[:,data.labels==1])
    # pearsonsCorrelationCoefficient = computeCorrelationMatrix(data.samples[:,data.labels==0])
    # plot.heatMap(pearsonsCorrelationCoefficient, classs=0)
    # plot.scatter(data.samples, data.labels)
    # plot.histogram(data.samples, data.labels)

#     ===================================================---- MVG classifiers <Validation> ----=========================================================
#     model = "G"# 0.1134
# #     model = "T"#0.1109
# #     model = "N"#0.466
#     mvg = MVG(kfold=kfold.allFolds, model=model, zNorm=False, pca= True, pcaM= 11)
#     # measuring predictions
#     # plot.roc_curve(scores=llr, groundTruthLabels=mvg.groundTruthLabels, workingPoint=wp, model=model)
#     predictions = ((mvg.logLiklihoodRatio > wp.effectiveThresshold) * 1)
#     print()
#     mpGaussian = MeasuringPredictions(ground_truth_labels=mvg.groundTruthLabels, predicted_labels=predictions, fnC=wp.fnC, fpC=wp.fpC, target_prior=wp.target_prior)
#     minDCF, threshold = min_DCF(llr=mvg.logLiklihoodRatio, groundTruthLabels=mvg.groundTruthLabels, workingPoint=wp)
#     print(mpGaussian.cm)
#     print(f"normalDCF MVG={mpGaussian.normalDCF}")
#     print(f"minDCF MVG={minDCF}")
#     plot.drawBayesErrorPlot(scores=mvg.logLiklihoodRatio, groundTruthLabels=mvg.groundTruthLabels)
#
#     ==============================================---- Logistic Regression classifer <Validation> ----===================================================
#
#     lambdaa = 1.e-4
#     thilds = [0.5, 0.1, 0.9]
# #     model = "LR"
#     lr = LogisticRegression(kfold=kfold.allFolds, workingPoint=wp, lambdaa=lambdaa, pca=True, zNorm=False, pcaM=10)
# #     # plot.roc_curve(scores=lr.logLiklihoodRatio, groundTruthLabels=lr.groundTruthLabels, workingPoint=wp, model=model)
#     for thild in thilds:
#         effective = WorkingPoint(target_prior=thild, fnC=1.0, fpC=1.0)
#         minDCF, threshold = min_DCF(llr=lr.logLiklihoodRatio, groundTruthLabels=lr.groundTruthLabels, workingPoint=effective)
#         print(f"minDCF logistic={minDCF}")
#     plot.error_plot_lambda_rang_lr(allFolds=kfold.allFolds, workingPoint=wp, pca=False, zNorm=False, pcaM=11, title= "error plot for lambda with different values(ballanced) both validaton and evaluatoin")

#    =============================================================---- SVM classifiers ----================================================================

    rbf = False
    poly = False
    gammaa = numpy.exp(-5)
    # trget = None
    trget = wp.piTilde
    # svm = SupportVectorMachine(kfold=kfold.allFolds, rbf=rbf, polynomial=poly, targetPrior=trget, C=0.1, gamma=gammaa, pca=False, zNorm=False, pcaM=12)
    # thilds = [0.5, 0.1, 0.9]
    # for thilde in thilds:
    #     print(thilde)
    #     wp2= WorkingPoint(target_prior=thilde, fpC=1.0, fnC=1.0)
    #     print(f"minDCF of un calibrated svm {min_DCF(svm.allScores, svm.groundTrouthLabels, workingPoint=wp2)[0]}")
    plot.error_plot_c_range_linear_and_poly_svm_wrapper(allFolds=kfold.allFolds, trget=trget, rbf=rbf, poly=poly, pca=False, pcaM=12, zNorm=False)

#    ==========================================================----Gaussian Mixture Model ----==============================================================

  # model= "full"
    # model= "diagnal"
    # # model= "tied"
    # gmm = GMM(kfold=kfold.allFolds, thresholdForEValues= 0.01, model=model, numberOfComponents=4, pca=False, pcaM=11, zNorm=False)
    # print(f"minDCF of un calibrated gmm {min_DCF(gmm.logLiklihoodRatio, gmm.groundTruthLabels, workingPoint=wp)[0]}")

    # plot.error_plot_gmm_different_components(allFolds=kfold.allFolds, piTilde=wp.piTilde, title="min DCF", rangeComponents=[1,2,4,8,16])
    # plot.error_plot_gmm_different_components(title="GMM With Different Components", model="tied model")


    # data_without_PCA_tied = "tied/gmm_components_4_pca_False_11_zNorm_False"
    # label_without_PCA_tied = "tied/label_gmm_components_4_pca_False_11_zNorm_False"
    # #
    # data_without_PCA_full = "full/gmm_components_4_pca_False_11_zNorm_False"
    # label_without_PCA_full = "full/label_gmm_components_4_pca_False_11_zNorm_False"
    #
    # with open(os.getcwd() + '/result/gmm/eval/' + data_without_PCA_full, 'rb') as f:
    #     gmmData = pickle.load(f)
    #
    # with open(os.getcwd() + '/result/gmm/eval/' + label_without_PCA_full, 'rb') as f:
    #     gmmLabel = pickle.load(f)

    #
    # with open(os.getcwd() + '/result/gmm/' + data_without_PCA_full, 'rb') as f:
    #     data = pickle.load(f)
    #
    # with open(os.getcwd() + '/result/gmm/' + label_without_PCA_full, 'rb') as f:
    #     label = pickle.load(f)

    # thilds = [0.5, 0.1, 0.9]
    # for thilde in thilds:
    #     print(thilde)
    #     wp2= WorkingPoint(target_prior=thilde, fpC=1.0, fnC=1.0)
    #     print(f"minDCF of un calibrated gmm {min_DCF(gmmData, gmmLabel, workingPoint=wp2)[0]}")


# ------------------------------------------------------------ Check Calibration ------------------------------------------------------------
#     data_without_PCA_tied = "tied/gmm_components_4_pca_False_12_zNorm_False"
#     label_without_PCA_tied = "tied/label_gmm_components_4_pca_False_12_zNorm_False"
#
#     with open(os.getcwd() + '/result/gmm/' + data_without_PCA_tied, 'rb') as f:
#         gmmData = pickle.load(f)
#
#     with open(os.getcwd() + '/result/gmm/' + label_without_PCA_tied, 'rb') as f:
#         gmmLabel = pickle.load(f)
#
#     kfoldCalibration = KFold(samples=toRow(gmmData), labels=gmmLabel, n_splits=3, seed=70)
#     calibratedGMM = LogisticRegression(kfold=kfoldCalibration.allFolds, workingPoint=wp, lambdaa=1e-4, pca=False, zNorm=False)
#
#     thilds = [0.5, 0.1, 0.9]
#     for piThilde in thilds:
#         logOdds = -numpy.log(piThilde/(1-piThilde))
#         # predictions = ((gmmData > logOdds) * 1)
#         predictions = ((calibratedGMM.logLiklihoodRatio > logOdds) * 1)
#         print(piThilde)
#         wp2 = WorkingPoint(target_prior=piThilde, fpC=1.0, fnC=1.0)
#         # mpGMMTied = MeasuringPredictions(ground_truth_labels=gmmLabel, predicted_labels=predictions, fnC=wp.fnC, fpC=wp.fpC, target_prior=piThilde)
#         mpGMMTied = MeasuringPredictions(ground_truth_labels=calibratedGMM.groundTruthLabels, predicted_labels=predictions, fnC=wp2.fnC, fpC=wp2.fpC, target_prior=piThilde)
#         print(mpGMMTied.normalDCF)
#         print(f"minDCF of un calibrated gmm {min_DCF(calibratedGMM.logLiklihoodRatio, calibratedGMM.groundTruthLabels, workingPoint=wp2)[0]}")
#
#
#
#
#     print("---------------------------------------------------------------------------------------------------------------")
#     svmRBF = "svm_rbfTrue_gamma_0.006737946999085467_poly_False_pca_False_12_zNorm_False_balanced_0.5"
#     with open(os.getcwd() + '/result/svm/' + svmRBF, 'rb') as f:
#         svmRBFData = pickle.load(f)
#
#     svmRBFL = "label_svm_rbfTrue_gamma_0.006737946999085467_poly_False_pca_False_12_zNorm_False_balanced_0.5"
#     with open(os.getcwd() + '/result/svm/' + svmRBFL, 'rb') as f:
#         svmRBFlabel = pickle.load(f)
#
#     kfoldCalibration = KFold(samples=toRow(svmRBFData), labels=svmRBFlabel, n_splits=3, seed=70)
#     calibratedSVM = LogisticRegression(kfold=kfoldCalibration.allFolds, workingPoint=wp, lambdaa=1e-4, pca=False, zNorm=False)
#
#     # plot.bayesErrorPlot([calibratedGMM.logLiklihoodRatio, calibratedSVM.logLiklihoodRatio], [calibratedGMM.groundTruthLabels, calibratedSVM.groundTruthLabels], title="display calibrated scores",model=["GMM","SVM"])
#     # plot.bayesErrorPlot([calibratedGMM.logLiklihoodRatio, calibratedSVM.logLiklihoodRatio], [calibratedGMM.groundTruthLabels, calibratedSVM.groundTruthLabels], title="display mis-calibration",model=["GMM","SVM"])
#
#     thilds = [0.5, 0.1, 0.9]
#     for piThilde in thilds:
#         logOdds = -numpy.log(piThilde / (1 - piThilde))
#         # predictions = ((gmmData > logOdds) * 1)
#         predictions = ((calibratedSVM.logLiklihoodRatio > logOdds) * 1)
#         print(piThilde)
#         wp2 = WorkingPoint(target_prior=piThilde, fpC=1.0, fnC=1.0)
#         # mpGMMTied = MeasuringPredictions(ground_truth_labels=gmmLabel, predicted_labels=predictions, fnC=wp.fnC, fpC=wp.fpC, target_prior=piThilde)
#         mpGMMTied = MeasuringPredictions(ground_truth_labels=calibratedSVM.groundTruthLabels,
#                                          predicted_labels=predictions, fnC=wp2.fnC, fpC=wp2.fpC, target_prior=piThilde)
#         print(mpGMMTied.normalDCF)
#         print(
#             f"minDCF of un calibrated svm {min_DCF(calibratedSVM.logLiklihoodRatio, calibratedSVM.groundTruthLabels, workingPoint=wp2)[0]}")

    # thilds = [0.5, 0.1, 0.9]
    # for piThilde in thilds:
    #     logOdds = -numpy.log(piThilde / (1 - piThilde))
    #     # predictions = ((svmRBFData > logOdds) * 1)
    #     predictions = ((calibratedSVM.logLiklihoodRatio > logOdds) * 1)
    #     print(piThilde)
    #     mpSVMRBF = MeasuringPredictions(ground_truth_labels=calibratedSVM.groundTruthLabels, predicted_labels=predictions, fnC=wp.fnC, fpC=wp.fpC, target_prior=piThilde)
    #     print(mpSVMRBF.normalDCF)
#


# --------------------------------------------------------------- ROC & erro_bayes_plot --------------------------------------------
#     data_without_PCA_tied = "tied/gmm_components_4_pca_False_11_zNorm_False"
#     label_without_PCA_tied = "tied/label_gmm_components_4_pca_False_11_zNorm_False"
#
#     # data_without_PCA_full = "gmm_components_4_pca_False_12_zNorm_False"
#     # label_without_PCA_full = "label_gmm_components_4_pca_False_12_zNorm_False"
#
#     with open(os.getcwd() + '/result/gmm/eval/' + data_without_PCA_tied, 'rb') as f:
#         gmmData = pickle.load(f)
#
#     with open(os.getcwd() + '/result/gmm/eval/' + label_without_PCA_tied, 'rb') as f:
#         gmmLabel = pickle.load(f)
#
#     svmRBF = "svm_rbfTrue_gamma_0.006737946999085467_poly_False_pca_False_12_zNorm_False_balanced_0.5"
#     with open(os.getcwd() + '/result/svm/eval/' + svmRBF, 'rb') as f:
#         svmRBFData = pickle.load(f)
#
#     svmRBFL = "label_svm_rbfTrue_gamma_0.006737946999085467_poly_False_pca_False_12_zNorm_False_balanced_0.5"
#     with open(os.getcwd() + '/result/svm/eval/' + svmRBFL, 'rb') as f:
#         svmRBFlabel = pickle.load(f)
#
#     kfoldCalibrationSVM = KFold(samples=toRow(svmRBFData), labels=svmRBFlabel, n_splits=3, seed=70)
#     calibratedSVM = LogisticRegression(kfold=kfoldCalibrationSVM.allFolds, workingPoint=wp, lambdaa=1e-4, pca=False, zNorm=False)
#
#     kfoldCalibrationGMM = KFold(samples=toRow(gmmData), labels=gmmLabel, n_splits=3, seed=70)
#     calibratedGMM = LogisticRegression(kfold=kfoldCalibrationGMM.allFolds, workingPoint=wp, lambdaa=1e-4, pca=False, zNorm=False)
# #
#     plot.roc_curve_of_best_models(scores=[svmRBFData, gmmData], groundTruthLabels= [svmRBFlabel, gmmLabel],workingPoint=wp, model=["[Eval]SVM-RBF(C=1.0, log(\u03B3)=-5, $\u03C0_{T}=0.5$)","[Eval]GMM-Tied(4Components)"])
#     plot.bayesErrorPlot(scores=[svmRBFData, gmmData], groundTruthLabels= [svmRBFlabel, gmmLabel], title="evaluation bayes error plot", model=["[Eval]SVM-RBF","[Eval]GMM"])
#     plot.bayesErrorPlot(scores=[calibratedSVM.logLiklihoodRatio, calibratedGMM.logLiklihoodRatio], groundTruthLabels= [calibratedSVM.groundTruthLabels, calibratedGMM.groundTruthLabels], title="evaluation bayes error plot(calibrated scores)", model=["[Eval]SVM-RBF","[Eval]GMM"])
# #

# ------------------------------------------------------------------------------Evaluation-------------------------------------------------------------
            # ---------------------------------------MVG-------------------------------------
# #     model = "G"  # 0.1134
# #     model = "T"#0.1109
#     model = "N"#0.466
#     mvg = MVG(kfold=[((data.samples, data.labels),(dataTest.samples, dataTest.labels))], model=model, zNorm=False, pca= True, pcaM= 10)
#     # measuring predictions
#     minDCF, threshold = min_DCF(llr=mvg.logLiklihoodRatio, groundTruthLabels=mvg.groundTruthLabels, workingPoint=wp)
#     print(f"minDCF MVG={minDCF}")
            # ---------------------------------------LR-------------------------------------

#     lambdaa = 1.e-4
#     thilds = [0.5, 0.1, 0.9]
# #     model = "LR"
#     lr = LogisticRegression(kfold=[((data.samples, data.labels),(dataTest.samples, dataTest.labels))], workingPoint=wp, lambdaa=lambdaa, pca=True, zNorm=False, pcaM=10)
# #     # plot.roc_curve(scores=lr.logLiklihoodRatio, groundTruthLabels=lr.groundTruthLabels, workingPoint=wp, model=model)
#     for thild in thilds:
#         effective = WorkingPoint(target_prior=thild, fnC=1.0, fpC=1.0)
#         # predictions = (lr.logLiklihoodRatio > effective.effectiveThresshold) * 1
#         # mpLR = MeasuringPredictions(ground_truth_labels=lr.groundTruthLabels, predicted_labels=predictions, fnC=wp.fnC, fpC=wp.fpC, target_prior=thild)
#         # print(mpLR.cm)
#         print(f"tilde={thild}")
#         # print(f"normalDCF logistic={mpLR.normalDCF}")
#         minDCF, threshold = min_DCF(llr=lr.logLiklihoodRatio, groundTruthLabels=lr.groundTruthLabels, workingPoint=effective)
#         print(f"minDCF logistic={minDCF}")
#         print("-------------------------------------------------------------------------------")

            # ---------------------------------------SVM-------------------------------------

    # rbf = True
    # poly = False
    # gammaa = numpy.exp(-5)
    # # trget = None
    # trget = wp.piTilde
    # svm = SupportVectorMachine(kfold=[((data.samples, data.labels),(dataTest.samples, dataTest.labels))], rbf=rbf, polynomial=poly, targetPrior=trget, C=1.0, gamma=gammaa, pca=False, zNorm=False, pcaM=12, eval=True)
    # thilds = [0.5, 0.1, 0.9]
    # for thilde in thilds:
    #     print(thilde)
    #     wp2= WorkingPoint(target_prior=thilde, fpC=1.0, fnC=1.0)
    #     print(f"minDCF of un calibrated svm {min_DCF(svm.allScores, svm.groundTrouthLabels, workingPoint=wp2)[0]}")

            # ---------------------------------------GMM-------------------------------------

    # model= "full"
    # # model= "diagnal"
    # # model= "tied"
    # gmm = GMM(kfold=[((data.samples, data.labels),(dataTest.samples, dataTest.labels))], thresholdForEValues= 0.01, model=model, numberOfComponents=1, pca=False, pcaM=11, zNorm=False, eval=True)
    # predictions = (gmm.logLiklihoodRatio > wp.effectiveThresshold) * 1
    # mpGMM = MeasuringPredictions(ground_truth_labels=gmm.groundTruthLabels, predicted_labels=predictions, fnC=wp.fnC,
    #                              fpC=wp.fpC, target_prior=wp.target_prior)
    # print(mpGMM.cm)
    # print(f"minDCF of un calibrated gmm {min_DCF(gmm.logLiklihoodRatio, gmm.groundTruthLabels, workingPoint=wp)[0]}")
    # plot.error_plot_gmm_different_components(title="GMM With Different Components - evaluation ", model="full model")


if __name__ == '__main__':
    main()