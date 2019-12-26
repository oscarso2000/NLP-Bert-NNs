import torch
import csv
import torch.optim as optim
import random
import pickle
import math 
import os

from tqdm import tqdm
import time
import torch.nn as nn
import transformers
from transformers import *

number_of_epochs = 10000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


CACHED = False

class FFNN(nn.Module):
	def __init__(self, input_dim, h):
			super(FFNN, self).__init__()
			self.h = h
			self.W1 = nn.Linear(input_dim, h)
			self.activation = nn.LeakyReLU()
			self.W2 = nn.Linear(h, 2)
   
			# The below two lines are not a source for an error
			self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
			self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)

	def forward(self, input_vector):
		# The z_i are just there to record intermediary computations for your clarity
		z1 = self.activation(self.W1(input_vector)) ##ERROR? ADDED ACTIVATION
		z2 = self.activation(self.W2(z1))
		predicted_vector = self.softmax(z2)
		return predicted_vector

def load_data(tokenizer, model):
	torch.set_grad_enabled(False)
	train_data = []
	with open("train.csv", newline='') as f:
		reader = csv.reader(f, dialect='excel')
		for i, row in enumerate(tqdm(reader)):
			e = row
			if e[7] is "1":
				x = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[5], add_special_tokens=True)])
				y = model(x)[1][0]
				train_data.append((y,1))

				#x = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[6], add_special_tokens=True)])
				#y = model(x)[1][0]
				#train_data.append((y,0))
			else:
				x = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[5], add_special_tokens=True)])
				y = model(x)[1][0]
				train_data.append((y,0))

				#x = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[6], add_special_tokens=True)])
				#y = model(x)[1][0]
				#train_data.append((y,1))


	valid_data = []
	with open("dev.csv", newline='') as f:
		reader = csv.reader(f, dialect='excel')
		for i, row in enumerate(tqdm(reader)):
			e = row
			if e[7] is "1":
				x = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[5], add_special_tokens=True)])
				y = model(x)[1][0]
				valid_data.append((y,1))

				#x = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[6], add_special_tokens=True)])
				#y = model(x)[1][0]
				#valid_data.append((y,0))
			else:
				x = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[5], add_special_tokens=True)])
				y = model(x)[1][0]
				valid_data.append((y,0))

				#x = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[6], add_special_tokens=True)])
				#y = model(x)[1][0]
				#valid_data.append((y,1))

	test_data = []
	with open("test.csv", newline='') as f:
		reader = csv.reader(f, dialect='excel')
		for i, row in enumerate(tqdm(reader)):
			e = row
			x1 = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[5], add_special_tokens=True)])
			x2 = torch.tensor([tokenizer.encode(e[1]+e[2]+e[3]+e[4], text_pair=e[6], add_special_tokens=True)])

			y1 = model(x1)[1][0]
			y2 = model(x2)[1][0]

			test_data.append( (e[0], y1, y2) )

	train_data = train_data[1:]
	valid_data = valid_data[1:]
	test_data  = test_data[1:]
	torch.set_grad_enabled(True)
	## TODO load valid and test set.
	return (train_data, valid_data, test_data)

			
def dummy_data():
	train_data = []
	valid_data = []
	test_data  = []
	for i in range(1000):
		x1 = random.randrange(0, 2)
		x2 = random.randrange(0, 2)
		train_data.append( (torch.tensor([1.0*x1, 1.0*x2]), x1^x2) )

	for i in range( 300 ):
		x1 = random.randrange(0, 2)
		x2 = random.randrange(0, 2)
		valid_data.append( (torch.tensor([1.0*x1, 1.0*x2]), x1^x2) )

	for i in range( 1500 ):
		x1 = random.randrange(0, 2)
		x2 = random.randrange(0, 2)
		test_data.append( (torch.tensor([1.0*x1, 1.0*x2]), x1^x2) )

	return (train_data, valid_data, test_data)

def write_output( output, filename ):
	with open( filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, dialect='excel')
		writer.writerow(["Id", "Prediction"])

		for line in output:
			writer.writerow(line)
def main():
	try:
		tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer/')
	except:
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		tokenizer.save_pretrained('./bert_tokenizer/')

	try:
		model = BertModel.from_pretrained('./bert_model/')
	except:
		model = BertModel.from_pretrained('bert-base-uncased')
		model.save_pretrained('./bert_model/')

	
	
	if CACHED:
		train_data, valid_data, test_data = pickle.load( open( "data.pkl", "rb" ) )
	else:
		train_data, valid_data, test_data = load_data(tokenizer, model)
		pickle.dump( (train_data, valid_data, test_data), open( "data.pkl", "wb" ) )

	## Train a FFNN to be the last layer between the 768 dimensional embeddings and the output

	ffnn = FFNN(input_dim = 768, h=32)
	optimizer = optim.SGD(ffnn.parameters(),lr=0.1, momentum=0.9)

	## TRAIN
	print("Training for {} epochs".format(number_of_epochs))
	for epoch in range(number_of_epochs):
		ffnn.train()
		optimizer.zero_grad()
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		print("Training started for epoch {}".format(epoch + 1))
		random.shuffle(train_data) # Good practice to shuffle order of training data
		minibatch_size = 16 
		N = len(train_data) 
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad() #they added this: fourth error
			loss = None
			for example_index in range(minibatch_size):
				input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
				predicted_vector = ffnn(input_vector)
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
				example_loss = ffnn.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / minibatch_size
			if minibatch_size is 0:
				print(loss)
			#loss.backward(retain_graph=True)
			loss.backward()
			optimizer.step()
		print("Training completed for epoch {}".format(epoch + 1))
		print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Training time for this epoch: {}".format(time.time() - start_time))

		## VALIDATION 

		ffnn.eval() #ERROR 2 ADDED: if not, entire model would continuously be training and not testing validation set
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		print("Validation started for epoch {}".format(epoch + 1))
		random.shuffle(valid_data) # Good practice to shuffle order of training data #ERROR: RANDOM SHUFFLE VALID_DATA!!!!!
		minibatch_size = 16 
		N = len(valid_data) ##ERROR: VALID DATA NOT TRAINDATA
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad() #they added this 
			loss = None
			for example_index in range(minibatch_size):
				input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index] ##ERROR: VALID_DATA
				predicted_vector = ffnn(input_vector)
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
				example_loss = ffnn.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / minibatch_size
			loss.backward(retain_graph=True)
			optimizer.step()
		print("Validation completed for epoch {}".format(epoch + 1))
		print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))

		if correct / total > 0.75:
			break

	
	output = []
	for example in test_data:
		result1 = ffnn(example[1])
		result2 = ffnn(example[2])

		if result1[1].item() > result2[1].item():
			output.append( [example[0], 1 ])
		else:
			output.append( [example[0], 2 ])

	write_output(output, "out.csv" )
	
	## COMPARE TEST EXAMPLES

main()

