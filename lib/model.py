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

		self.qlstm = nn.LSTM(1,1,batch_first=True,bidirectional=True)
		self.clstm = nn.LSTM(1,1,batch_first=True,bidirectional=True)
		#self.hidden = torch.zeros(2,).to(self.device)
		#self.cell = torch.zeros().to(self.device)
		self.fc = nn.Linear(8,1)

	def forward(self, inputData):
		queryTensor,contextTensor,tagTensor = inputData

		query,_ = self.lstm(queryTensor.unsqueeze(0)) # len * dim
		context,_ = self.lstm(contextTensor.unsqueeze(0)) # len * dim

		queryTensorLen = query.size(1)
		contextTensorLen = context.size(1)
		query = query.squeeze().unsqueeze(2).expand(-1,-1,contextTensorLen)
		context = context.squeeze().transpose(0,1).unsqueeze(0).expand(queryTensorLen,-1,-1)

		cos = nn.CosineSimilarity(dim=1)
		simMat = cos(query,context)
		queryLenMat = torch.max(simMat,1)[0]
		contextLenMat = torch.max(simMat,0)[0]
		
		querySim = queryLenMat.unsqueeze(1).unsqueeze(0)
		contextSim = contextLenMat.unsqueeze(1).unsqueeze(0)
		querySim,_ = self.qlstm(querySim)
		contextSim,_ = self.clstm(contextSim)

		querySim = torch.cat((querySim.squeeze()[0],querySim.squeeze()[-1]),0)
		contextSim = torch.cat((contextSim.squeeze()[0],contextSim.squeeze()[-1]),0)

		wholeSim = torch.cat((querySim,contextSim),0)
		sig = nn.Sigmoid()
		finalScore = sig(self.fc(wholeSim))
		return finalScore


