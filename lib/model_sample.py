import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		hiddenDim = 400
		embedDim = 300
		self.hiddenDim = hiddenDim
		self.embedDim = embedDim
#		nn.Module.__init__(self) #why it exists? : superclass initialization. same as line below.
		super(Net,self).__init__()
		self.device = torch.device("cuda:0")
		self.lstm = nn.LSTM(self.embedDim,self.hiddenDim,batch_first=True,bidirectional=True)
		#self.hidden = torch.zeros(2,).to(self.device)
		#self.cell = torch.zeros().to(self.device)
		self.fc = nn.Linear(self.hiddenDim*2,1)

	def forward(self, inputPair):
		dataTensor, resultTensor, lengthList = inputPair
		dataSeq = torch.nn.utils.rnn.pack_padded_sequence(dataTensor,lengthList,batch_first=True)
		lstmOutput,(hidden,cell) = self.lstm(dataSeq)
		seqResult, _ = torch.nn.utils.rnn.pad_packed_sequence(lstmOutput,batch_first=True)
		sentenceEmbed = torch.max(seqResult,1)[0].squeeze()
		sig = nn.Sigmoid()
		finalResult = sig(self.fc(sentenceEmbed)).squeeze()
		
		#print(finalResult)
		#print(finalResult.size())
		return finalResult
