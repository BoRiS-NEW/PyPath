"""
Correlate columns of a CSV file
2021-01-03
"""
import csv
import os,stat
import statistics
import math
import seaborn as sn
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import rankdata
from scipy.stats import norm
from math import sqrt
from types import resolve_bases
import numpy as np
import matplotlib as mptlib
from matplotlib import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
import tkinter.font as tkfnt
import pandas as pd
from pandas import DataFrame
from factor_analyzer import FactorAnalyzer
import array as ary
import scipy.stats as sps
from scipy.stats import chi2_contingency 
import matplotlib.widgets as wid
from matplotlib.widgets import RadioButtons, Button
from scipy.stats import rankdata
from tkinter import *
from tkinter import messagebox as mBox
from tkinter.filedialog import * 
from tkinter.simpledialog import * 
from tkinter import ttk
from datetime import date
nowDate = date.today().year.__str__()
myCopyright = '© G J Boris Allan, ' + str(nowDate)
root=tk.Tk()
root.option_add('*Font', 'Consolas')
mptlib.rcParams['font.sans-serif'] = ['Candara']
crlf = '\n'
aTab = '\t'
clrP = 'pink'
clrY = 'yellow' 
clrLy = 'lightyellow' 
clrPgr = 'palegoldenrod'
clrLb = 'lightblue'
arrow = '<='
annotFont=8
labelFont=10
titleFont= 14
decPlaces = '{:>15.3f}'
txtRight = '{:>15}'
txtLeft = '{:<15}'
#
inFileName = askopenfilename(title = 'Choose a delimited input file', filetypes=[('CSV files', '*.csv'), ('Text files', '*.txt')])
delim = askstring(title='Delimiter',prompt='What is the delimiter for the input file?', initialvalue=',')[0]
colHeadings = mBox.askquestion('Column headings', message='Do you have column headings?')
rowHeadings = mBox.askquestion('Row headings', message='Do you have row headings?')
labelFileName = ''
labelDf = pd.DataFrame()
if (colHeadings == 'yes'):
    hasColHead = 0
    labelFileName= askopenfilename(title = 'Choose a delimited label file or "Cancel" if no file', filetypes=[('CSV files', '*.csv'), ('Text files', '*.txt')])
else:
    hasColHead = None
if (rowHeadings == 'yes'):
    hasRowHead = 0
else:
    hasRowHead = None
df = pd.read_csv(inFileName, index_col=hasRowHead, header=hasColHead, delimiter=delim)
if (hasColHead != None):
    labelDf = pd.read_csv(labelFileName, header=hasColHead, delimiter=delim)
labelCols = labelDf.columns.to_list()
#
outFileName =asksaveasfilename(title = 'Choose a text output file, or type in a new text file name', filetypes = [('Text files', '*.txt')], defaultextension = '.txt')
outFile = open(outFileName, 'w+', buffering=1)
outPath = os.path.dirname(outFileName)+'/'
dfColumns = []
if (hasColHead == None):
    for iCol in df.columns:
        dfColumns.append('Var'+format(iCol, '03d'))
    df.columns = dfColumns
else:
    dfColumns = df.columns
dfRank=DataFrame(columns=dfColumns, index=df.index)
#
rankZ=[]
indexLen = len(df.index)
for I in range(0,indexLen):
    probZ = norm.ppf((I+0.5)/indexLen)
    rankZ.append(probZ)
for iCol in dfColumns:
    ranking = df[iCol].rank(ascending=1)
    dfRank[iCol]= ranking
#
outFile.write(crlf)
dfCorrPm = df.corr(method='pearson')
outFile.write('Pearson R')
outFile.write(crlf)
outFile.write(str(dfCorrPm))
outFile.write(crlf)
outFile.write(crlf)
dfCorrRho = df.corr(method='spearman')
outFile.write('Spearman Rho')
outFile.write(crlf)
outFile.write(str(dfCorrRho))
outFile.write(crlf)
#
toContinue = mBox.askyesno(title='Generate models',message='Do you want to generate path models?')
modelNum = 0
computePred = False
if toContinue:
    computePred = mBox.askyesno('Predicted values','Do you want to display predicted values from models?')
outFile.write(crlf+'===================='+crlf)
while (toContinue): 
# 
    modelNum += 1
    modelNumStr = format(modelNum, '02d')      
    modelStr = 'Model_' + modelNumStr
#
    winBox = Tk()  
    winBox.option_add('*Font', 'Consolas')    
    lbl = Label(winBox,text = 'Select the dependent variable', bd=5, padx=5, pady=5)  
    listbox = Listbox(winBox, bg=clrY, bd=5)     
    selOne = []   
    iList = 0 
    for iCol in df.columns:
        listbox.insert(iList,iCol) 
        iList += 1
    lbl.pack()  
    listbox.pack()  
    def selectDep():
        selOne.append(listbox.get(ANCHOR))
        winBox.quit() 
    selDep = tk.Button(winBox, text='Confirm dependent variable choice', command=selectDep, bg=clrP, bd=10, padx=10, pady=10)
    selDep.pack()
    winBox.mainloop()
    winBox.destroy()
    depVar = selOne [0]
#
    winBoxI = Tk()  
    winBoxI.option_add('*Font', 'Consolas')
    lbl = Label(winBoxI,text = "Select the independent variables", bd=5, padx=5, pady=5)  
    listbox = Listbox(winBoxI, selectmode=MULTIPLE, bg=clrY, bd=5) 
    selMany = []
    iList = 0 
    for iCol in df.columns:    
        if iCol != selOne[0]:
            listbox.insert(iList,iCol) 
        iList += 1
    lbl.pack()  
    listbox.pack()  
    def selectInd():
        selList = listbox.curselection() 
        for I in selList:
            selMany.append(listbox.get(I))        
        winBoxI.quit() 
    selInd = tk.Button(winBoxI, text='Confirm independent variable choices', command=selectInd, bg=clrP, bd=10, padx=10, pady=10)
    selInd.pack()
    winBoxI.mainloop()
    winBoxI.destroy()
    indVars = selMany
#
    totalVars = selOne+indVars
    dfTotalSubset = dfCorrRho.loc[totalVars,totalVars].copy()
    varExpl = np.linalg.inv(dfTotalSubset)[0][0]
    varExpl = (1-(1/(varExpl)))
    dfIndSubset = dfCorrRho.loc[indVars,indVars].copy()
    dfIndInvSubset = np.linalg.inv(dfIndSubset)
    indCorr = dfTotalSubset.loc[indVars,[depVar]]
#
    outFile.write(crlf+'----------'+crlf+modelStr+': '+str(selOne)+' '+arrow+' '+str(indVars))
    outFile.write(crlf+'----------'+crlf)
    pathCoeff = np.dot(dfIndInvSubset,indCorr)
    outFile.write(txtLeft.format(str(selOne))+txtRight.format('Path coeffs'))
    outFile.write(crlf)
    for I in range(len(indVars)):
        outFile.write(txtLeft.format(indVars[I])+decPlaces.format(float((pathCoeff[I][0]))))
        outFile.write(crlf)
    outFile.write('----------'+crlf)
    outFile.write(depVar+' variance explained: '+str(varExpl) + ' {'+ str(sqrt(varExpl))+'}')    
    outFile.write(crlf+'----------'+crlf)
    predVal =[0]*indexLen
    for iCol in range(len(indVars)):
        for jRow in range(0,indexLen):           
            predVal[jRow] += pathCoeff[iCol][0] * (dfRank.loc[df.index[jRow],indVars[iCol]] - (indexLen+1)/2)
    predVal = rankdata(predVal)
    origVal = []
    origVal = (dfRank[depVar]).to_numpy(copy=True)
    rho, p = spearmanr(predVal, origVal)
    outFile.write(txtLeft.format('Spearman Rho for predicted '+depVar+':')+decPlaces.format(rho) +crlf+txtLeft.format('Probability')+decPlaces.format(p)+')')
    outFile.write(crlf+'----------'+crlf)
#
    if computePred:
        fig = plt.figure()
        plt.figure(figsize=(7,7),
           facecolor=clrLy)
        plt.suptitle(depVar + ': Predicted rank vs observed rank', fontsize=titleFont,
                weight='bold', bbox=dict(boxstyle='square', facecolor='none', edgecolor='black'))
        plt.figtext(0.98, 0.02, myCopyright, horizontalalignment='right', size=6, weight='light')
        plt.xlabel(depVar + ' observed rank - 1 is lowest', fontsize=labelFont)
        plt.ylabel(depVar + ' predicted rank - 1 is lowest', fontsize=labelFont) 
        x = origVal
        y = predVal
        for I in range(indexLen):
            plt.annotate(''+df.index[I], (x[I], y[I]), fontsize=annotFont)
        gridNum = askinteger('Scatterplot rows and columns','How many rows (same as columns)?')
        plt.plot([1,indexLen],[1,indexLen], color=clrP, linewidth=1)
        gridSep = (indexLen+1)/gridNum
        for I in range(1,gridNum):
            plt.axvline(x=gridSep*I, color=clrLb, linewidth=1)
            plt.axhline(y=gridSep*I, color=clrLb, linewidth=1)
        plt.plot(x,y, marker='.', markeredgecolor='black', color=clrPgr, linestyle='None')
        safeLabel = depVar.replace('/','_')
        plt.savefig(outPath+safeLabel+'_'+modelNumStr+'.png', dpi=300)
        plt.clf()
        plt.cla()
        plt.close()
    #
    outFile.write(crlf+'===================='+crlf)

    toContinue = mBox.askyesno(title='New model',message='Do you want to create another model?')

toContinue = mBox.askyesno(title='New scatterplot',message='Do you want to create a new scatterplot?')
def new_func(xTab):
    chi, p, dof, expected = chi2_contingency(xTab)
    return p,chi

while (toContinue):
    outFile.write(crlf+'----------'+crlf)
    winBoxX = Tk()  
    winBoxX.option_add('*Font', 'Consolas')
    lbl = Label(winBoxX,text = "Select the X-axis variable", bd=5, padx=5, pady=5)  
    listbox = Listbox(winBoxX, selectmode=SINGLE, bg=clrY, bd=5) 
    selX = []
    iList = 0 
    for iCol in df.columns:
        listbox.insert(iList,iCol) 
        iList += 1
    lbl.pack()  
    listbox.pack()  
    def selectScatter():
        selX.append(listbox.get(ANCHOR))        
        winBoxX.quit() 
    xAxis = tk.Button(winBoxX, text='Confirm X-axis choice for scatterplot', command=selectScatter, bg=clrP, bd=10, padx=10, pady=10)
    xAxis.pack()
    winBoxX.mainloop()
    winBoxX.destroy()    
    x=dfRank[selX].to_numpy(copy=True)
#
    winBoxY = Tk()  
    winBoxY.option_add('*Font', 'Consolas')
    lbl = Label(winBoxY,text = "Select the Y-axis variable", bd=5, padx=5, pady=5)  
    listbox = Listbox(winBoxY, selectmode=SINGLE, bg=clrY, bd=5) 
    selY = []
    iList = 0 
    for iCol in df.columns:
        listbox.insert(iList,iCol) 
        iList += 1
    lbl.pack()  
    listbox.pack()  
    def selectScatter():
        selY.append(listbox.get(ANCHOR))        
        winBoxY.quit() 
    yAxis = tk.Button(winBoxY, text='Confirm Y-axis choice for scatterplot', command=selectScatter, bg=clrP, bd=10, padx=10, pady=10)
    yAxis.pack()
    winBoxY.mainloop()
    winBoxY.destroy()
    y = selY    
    y=dfRank[selY].to_numpy(copy=True)
#
    labelX=labelDf.values[0][labelCols.index(selX[0])]
    labelY=labelDf.values[0][labelCols.index(selY[0])]
    fig = plt.figure()
    plt.figure(figsize=(7,7),
           facecolor=clrLy)
    plt.subplot(1,1,1)
    plt.suptitle(labelY + ' vs ' + labelX + '\n' + str(indexLen) + ' ' + labelDf.values[0][0] + ' rankings', 
            fontsize=titleFont, y=0.95, weight='bold')
    plt.figtext(0.98, 0.02, myCopyright, horizontalalignment='right', fontsize=annotFont, weight='light')
    plt.xlabel(labelX + ', ' + labelDf.values[0][0] + ' ranking', fontsize=labelFont, weight='bold')
    plt.ylabel(labelY + ', ' + labelDf.values[0][0] + ' ranking', fontsize=labelFont, weight='bold') 
    plt.xticks([1,(indexLen+1)/2,indexLen],['Low\n1','Mid','High\n'+str(indexLen)], weight='bold', fontsize=annotFont)
    plt.yticks([1,(indexLen+1)/2,indexLen],['1\nLow','Mid','High\n'+str(indexLen)], weight='bold', fontsize=annotFont)
    # ax.set_xticklabels(['Low','High'])
    for I in range(indexLen):
        plt.annotate(''+df.index[I], (x[I], y[I]), fontsize=annotFont)
#
    gridNum = askinteger('Scatterplot rows and columns','How many rows and columns)'+crlf+'(maximum of 6)?')
    if gridNum > 6:
        gridNum = 6
    posNeg = np.sign(dfCorrRho.loc[selX[0],selY[0]])
    if (posNeg > 0):
        plt.plot([1,indexLen],[1,indexLen], color=clrP, linewidth=1)
    elif (posNeg < 0):
        plt.plot([1,indexLen],[indexLen,1], color=clrP, linewidth=1)
    gridSep = (indexLen+1)/gridNum
    xCat = ['']*(indexLen)
    yCat = ['']*(indexLen)
    sepCat = [0]*(gridNum+1)
    sepCat[0] = 0
    valLabel = ['']*gridNum
    valLabelRev = ['']*gridNum
    sepCat[gridNum]=indexLen+1
    for I in range(gridNum):
        if I == 0:
            valLabel[I] = 'Low'
            valLabelRev[I] = 'High'
        elif I == gridNum-1:
            valLabel[I] = 'High'
            valLabelRev[I] = 'Low'
        else:
            valLabel[I] = str(I+1)
            valLabelRev[I] = str(gridNum-I)
    for I in range(1,gridNum):
        axisVal = gridSep*I
        plt.axvline(x=axisVal, color=clrLb, linewidth=1)
        plt.axhline(y=axisVal, color=clrLb, linewidth=1)
        sepCat[I] = axisVal
    for J in range(indexLen):
        for I in range(gridNum):
            if (y[J] > sepCat[I]) and (y[J] <= sepCat[I+1]):
                yCat[J] = I
            if (x[J] > sepCat[I]) and (x[J] <= sepCat[I+1]):
                xCat[J] = I
    xyData = {}
    xyData[selX[0]] = xCat
    xyData[selY[0]] = yCat
    dfCat = pd.DataFrame(xyData,columns=[selX[0],selY[0]],index=df.index)
    aList = []
    xTab = pd.crosstab(dfCat[selY[0]],dfCat[selX[0]])
#
    aList = xTab.axes
    cList = xTab.columns
    vList = xTab.values
    p, chi = new_func(xTab)
    xTabRev=xTab.sort_index(ascending=False)
#
    outFile.write(str(xTab.sort_index(ascending=False)))
    outFile.write(crlf+txtLeft.format(str('Chi-squared')) + decPlaces.format(chi)+crlf+txtLeft.format('Probability')+decPlaces.format(p))
    tau, p = kendalltau(xCat, yCat)
    outFile.write(crlf+txtLeft.format(str('Kendall Tau-B ')) + decPlaces.format(tau)+crlf+txtLeft.format('Probability')+decPlaces.format(p))
    plt.plot(x,y, marker='.', markeredgecolor='black', color=clrPgr, linestyle='None')  
    safeLabelX = selX[0].replace('/','_')
    safeLabelY = selY[0].replace('/','_')
    plt.savefig(outPath+safeLabelY+'_'+safeLabelX+'['+format(gridNum, '02d')+'].png', dpi=300)
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close()
#
    plt.figure(figsize=(4,2),
           facecolor=clrLy)
    plt.subplots_adjust(left=0.2,right=0.9, bottom=0.2, top=0.8)
    the_table=plt.table(xTabRev.values,rowLabels=valLabelRev,colLabels=valLabel,rowColours=[clrPgr]*gridNum,
            colColours=[clrPgr]*gridNum,loc='center',cellLoc='center', rowLoc='center')
    the_table.scale(1, 7/(1+gridNum))
    ax = plt.gca()
    plt.xticks([])
    plt.yticks([])
    plt.box(on=None)
    # print(the_table(row=0,column=0))
    plt.suptitle(labelY+' vs '+labelX+'\n'+str(indexLen)+ ' ' + labelDf.values[0][0], weight='bold', fontsize=labelFont, y=0.95, x=0.9, 
            ha='right') #,
        #    bbox=dict(boxstyle='square', facecolor='none', edgecolor='black'))
    plt.figtext(0.98, 0.02, myCopyright, horizontalalignment='right', size=6, weight='light')
    plt.figtext(0.05,0.2,labelY, rotation=90, va='bottom', ha='center', fontsize=annotFont, weight='bold')
    plt.figtext(0.55,0.1,labelX, rotation=0, va='center', ha='center', fontsize=annotFont, weight='bold')
    plt.draw()
    fig = plt.gcf()
    # plt.show()
    plt.savefig(outPath+safeLabelY+'_'+safeLabelX+'_Table.png', format='png', dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    toContinue = mBox.askyesno(title='New scatterplot',message='Do you want to create a new scatterplot?')