from re import A
from cmath import nan
import numpy as np
import pandas as pd
import time
import pandas_ta as pda
import os

nInst = 100
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos
    (_, date) = prcSoFar.shape
    # read log
    log = open("log.txt", "a")
    pastHist = readArray(date, "log.txt")
    if (date == 1):
        currentPos = np.zeros(100)
    else:
        currentPos = np.zeros(0)
        # generate RSI on date from prcSoFar
        prcAllExtract = prcSoFar[:, 0:date]
        prcAllExtractDF = pd.DataFrame(prcAllExtract)
        prcAllExtractDF = prcAllExtractDF.T
        rsi = generateRSIDF(prcAllExtractDF, date)
        (ma, std) = generateMADF(prcAllExtractDF, date, 14)
        for instrument in range(0, 100):
            instrumentPrice = prcSoFar[instrument, -1]
            instrumentRSI = rsi[instrument]
            instrumentMA = ma[instrument]
            instrumentSTD = std[instrument]

            # get yesterday positions
            pastPos = pastHist[-1, instrument]
            # print("past position at "+str(instrument)+" is: "+str(pastPos))

            # True means currently having position open
            if (pastPos != 0):
                positionOpen = True
            else:
                positionOpen = False

            # print("decision at "+str(instrument)+ " is "+str(positionOpen))
            instrumentDecision = decision(instrumentPrice, positionOpen, pastPos, instrumentRSI,
                                          instrumentMA, instrumentSTD, 30, 70)
            shareTrade = 500
            if (instrumentDecision == 1):
                currentPos = np.append(currentPos, shareTrade) # initial buy
            elif (instrumentDecision == 2):
                currentPos = np.append(currentPos, -shareTrade) # initial short
            elif (instrumentDecision == 3):
                currentPos = np.append(currentPos, 0) # close all position
            elif (instrumentDecision == 4):
                currentPos = np.append(currentPos, pastPos+shareTrade) # buy more
            elif (instrumentDecision == 5):
                currentPos = np.append(currentPos, pastPos-shareTrade) # short more
            elif (instrumentDecision == 0):
                currentPos = np.append(currentPos, pastPos) # holding the position

    # write a log to a file
    np.savetxt(log, currentPos)
    log.close()
    if (date==249):
        if os.path.isfile("log.txt"):
            os.remove("log.txt")
    return currentPos


def readArray(date, filename):
    logHistAll = np.loadtxt(filename)
    if (date > 1):
        logHistAll = np.reshape(logHistAll, (date-1, -1))
    return logHistAll


def decision(price, positionOpen, pastPos, rsi, ma, std, buySignalRSI, shortSignalRSI):
    if (positionOpen == False):
        # buy long
        # 0 mean hold, 1 mean buy, 3 sell to close
        if (price < ma-3*std):
            return 1  # long
        # sell short
        # 0 mean hold, 2 mean short, 3 mean buy to close
        elif (price > ma+3*std):
            return 2  # short
        else:
            return 0  # hold
    else:  # having an open position
        if (pastPos > 0):
            if (price > ma):
                return 3  # close position of long
            elif (price < ma-1*std):
                return 4  # buy more
            else:
                return 0
        elif (pastPos < 0):
            if (price < ma):
                return 3  # close position of short
            elif (price > ma+1*std):
                return 5  # short more
            else:
                return 0
        else:
            return 0  # hold


def generateRSIDF(prcAllExtractDF: pd.DataFrame, date: np.integer):
    rsiAllColumn = []
    for i in range(0, 100):
        rsiDFEach = prcAllExtractDF.iloc[:, i]
        rsi = pd.DataFrame(columns=["RSI"])
        rsi["RSI"] = pda.rsi(rsiDFEach, length=14)
        if (date > 14):
            x = rsi.at[date-1, "RSI"]
        else:
            x = np.NaN
        rsiAllColumn.append(x)
    return rsiAllColumn


def generateMADF(prcAllExtractDF: pd.DataFrame, date: np.integer, period: np.integer):
    maAllColumn = []
    stdAllColumn = []  # sample standard deviation
    for i in range(0, 100):
        maDFEach = prcAllExtractDF.iloc[:, i]
        ma = pd.DataFrame(columns=["MA", "STD"])
        ma["MA"] = pda.sma(maDFEach, length=period)
        ma["STD"] = pda.stdev(maDFEach, length=period)
        if (date >= period):
            y = ma.at[date-1, "MA"]
            z = ma.at[date-1, "STD"]
        else:
            y = np.NAN
            z = np.NAN
        maAllColumn.append(y)
        stdAllColumn.append(z)
    return (maAllColumn, stdAllColumn)
