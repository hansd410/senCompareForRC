from lib.readData import triviaData
from lib.model import Net
from lib.wordEmbed import Embedding

import torch.nn as nn
import torch.optim as optim

embedDim = 300
epoch = 2
learningRate = 0.0001

trainDataDir = "data/tenFold/train_0"
testDataDir = "data/tenFold/test_0"

# read data
wordEmbed = Embedding(embedDim)
trainData = triviaData(trainDataDir,wordEmbed)
testData = triviaData(testDataDir,wordEmbed)
trainDataCount = trainData.getDataNum()
testDataCount = testData.getDataNum()
# network init
net = Net().cuda()

fout = open("Result.txt",'w')

# batch size is 1 (fixed)
for i in range(epoch*trainDataCount):
	queryTensor,contextTensor,tagTensor = trainData.getData()
	result=net((queryTensor,contextTensor,tagTensor))
	loss = ((tagTensor-result)**2)
	optimizer = optim.Adam(net.parameters(),lr=learningRate)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if(i==0 or i%99==0):
		print("test begins at "+str(i)+"th iteration")
		trainLoss = 0
		for j in range(trainDataCount):
			queryTensor,contextTensor,tagTensor = trainData.getData()
			result=net((queryTensor,contextTensor,tagTensor))
			trainLoss += (tagTensor-result).item()**2
		trainLoss = trainLoss/trainDataCount
		print("trainLoss\t"+str(trainLoss))
		fout.write("trainLoss\t"+str(trainLoss)+"\n")

		tpCount = 0
		tnCount = 0
		fpCount = 0
		fnCount = 0
		testLoss = 0
		for j in range(testDataCount):
			queryTensor,contextTensor,tagTensor = testData.getData()
			result=net((queryTensor,contextTensor,tagTensor))
			testLoss += (tagTensor-result).item()**2
			if(tagTensor.item() ==0):
				if(result.item()>=0.5):
					fpCount += 1
				else:
					tnCount += 1
			else:
				if(result.item()>=0.5):
					tpCount += 1
				else:
					fnCount += 1
		testLoss = testLoss/testDataCount
		fout.write("testLoss\t"+str(testLoss)+"\n")
		fout.write("tp/fp/fn/tn\t"+str(tpCount)+"\t"+str(fpCount)+"\t"+str(fnCount)+"\t"+str(tnCount)+"\n")
		print("testLoss\t"+str(testLoss))
		print("tp/fp/fn/tn\t"+str(tpCount)+"\t"+str(fpCount)+"\t"+str(fnCount)+"\t"+str(tnCount)+"\n")

		if(fnCount+tnCount ==0):
			fout.write("false precision\t"+str(0)+"\n")
			print("false precision\t"+str(0)+"\n")
		else:
			fout.write("false precision\t"+str(tnCount/(fnCount+tnCount))+"\n")
			print("false precision\t"+str(tnCount/(fnCount+tnCount))+"\n")
