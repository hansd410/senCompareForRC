import re
import nltk
import torch
#from . wordEmbed import Embedding

class triviaData:
	def __init__(self,dataDir,wordEmbed):
		self.device = torch.device("cuda:0")
		embedDim=300
		#wordEmbed = Embedding(embedDim)
		self.wordEmbedding = wordEmbed.getEmbed()
		self.wordIdxDic = wordEmbed.getWordIdxDic()

		self.triviaDir = "../../data/triviaqa/evidence/wikipedia/"
		#self.fin = open("data/tenFold/test_0",'r')
		self.fin = open(dataDir,'r')

		self.dataList = []
		self.dataIdx = 0
		self.dataNum = 0
		
		print("read data")

		while(True):
			line = self.fin.readline().rstrip()
			if not line:
				break
			token = line.split('\t')
			qid = token[0]
			# query
			query = token[1]
			tag = token[2]
			evidenceStr = token[3]
			evidenceList = evidenceStr.split(", ")
			self.dataNum += 1

			evidenceFin = open(self.triviaDir+evidenceList[0],'r')
			# context
			evidenceDoc = evidenceFin.read()
			# preprocess
			query = query.lower().split(' ')
			for i in range(len(query)):
				if(query[i] not in self.wordIdxDic.keys()):
					query[i] = '<unk>'
			queryIdxs = [self.wordIdxDic[w] for w in query]
			queryIdxTensor = torch.LongTensor(queryIdxs)
			queryTensor = self.wordEmbedding(queryIdxTensor).to(self.device)

			evidenceDoc = evidenceDoc.lower().split(' ')
			for i in range(len(evidenceDoc)):
				if(evidenceDoc[i] not in self.wordIdxDic.keys()):
					evidenceDoc[i] = '<unk>'		
			evidenceIdxs = [self.wordIdxDic[w] for w in evidenceDoc]
			evidenceIdxs = evidenceIdxs[0:500]
			evidenceIdxTensor = torch.LongTensor(evidenceIdxs)
			evidenceTensor = self.wordEmbedding(evidenceIdxTensor).to(self.device)

			if(tag == "T"):
				tagTensor = torch.Tensor([1]).to(self.device)
			else:
				tagTensor = torch.Tensor([0]).to(self.device)

			self.dataList.append((queryTensor,evidenceTensor,tagTensor))

		print("data reading done")

	def preProcess(self,inputStr):
		inputStr = re.sub('\s',' ',inputStr)
		inputStr = " ".join(nltk.word_tokenize(inputStr))
		return inputStr

	def getData(self):
		dataIdx = self.dataIdx%self.dataNum
		returnData = self.dataList[dataIdx]

		self.dataIdx += 1
		return returnData

	def getDataNum(self):
		return self.dataNum
