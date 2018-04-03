#Program 2
import nltk 
from nltk.corpus import udhr
from nltk import bigrams
from nltk.util import ngrams
from nltk import FreqDist
from nltk import ConditionalFreqDist

#Creating a class for Language models to distinguish between words in different languages
class LangModel:

	def __init__(self,file):
		corpus = udhr.raw(file)
		self.training_set=corpus[0:1000]
		token= list(self.training_set)
		self.unigram=token
		self.bigram=list(nltk.bigrams(token))
		self.trigram=list(nltk.trigrams(token))
		self.unigram_frequency=FreqDist(self.unigram)
		self.bigram_frequency=ConditionalFreqDist(self.bigram)
		self.trigam_frequency=ConditionalFreqDist(list(((x,y),z) for x,y,z in self.trigram))

#Creating a function cal_unigram for calculating the probability of each character in Uniigram model
	def cal_unigram(self,words):
		words=words.strip().lower()
		character=list(words)
		p=1
		for n in character:
			p=p*self.unigram_frequency.freq(n)
		return p

#Creating a function cal_bigram for calculating the probability of each character in Bigram model		
	def cal_bigram(self,words):
		words= words.strip().lower()
		character=list(words)
		p=1
		for m,n in enumerate(character):
			if m == 0:
				continue
			p=p*self.bigram_frequency[character[m - 1]].freq(n)
		return p

#Creating a function cal_trigram for calculating the probability of each character in Trigram model			
	def cal_trigram(self,words):
		words=words.strip().lower()
		character=list(words)
		p=1
		for m, n in enumerate(character):
			if m <= 1:
				continue
			p=p*self.trigam_frequency[(character[m - 2], character[m - 1])].freq(n)
		return p

#Creating a function to calculate the Accuracy of unigram model, bigram model and trigram model and calculating the prediction of each word based on the character model 
def Cal_Pred_Acc(charmodel,chardataset):
	model = LangModel(charmodel)
	words = udhr.words(chardataset)[0:1000]
	word_count = len(words) #Calculating the total number of words in a set
	unigram_acc = 0
	bigram_acc = 0
	trigram_acc = 0
	
	for word in words:
		uni_pred = model.cal_unigram(word)
		if(uni_pred > 0):
			unigram_acc = unigram_acc + 1
		print("%15s - %19.18f" %(word,uni_pred))
	print("\nAccuracy of unigram model: ", unigram_acc * 100 / word_count,'%','\n')
	
	for word in words:
		bi_pred = model.cal_bigram(word)
		if(bi_pred > 0):
			bigram_acc = bigram_acc + 1
		print("%15s - %19.18f" %(word,bi_pred))
	print("\nAccuracy of bigram model: ", bigram_acc * 100 / word_count,'%','\n')
		
	for word in words:
		tri_pred = model.cal_trigram(word)
		if(tri_pred > 0):
			trigram_acc = trigram_acc + 1
		print("%15s - %19.18f" %(word,tri_pred))
	print("\nAccuracy of trigram model: ", trigram_acc * 100 / word_count,'%','\n')	
	
Cal_Pred_Acc('English-Latin1', 'English-Latin1') #To calculate prediction of each english word using english langugae model
Cal_Pred_Acc('English-Latin1', 'French_Francais-Latin1') #To calculate prediction of each french word using english langugae model
Cal_Pred_Acc('Spanish_Espanol-Latin1', 'Spanish_Espanol-Latin1') #To calculate prediction of each spanish word using spanish langugae model
Cal_Pred_Acc('Spanish_Espanol-Latin1', 'Italian_Italiano-Latin1') #To calculate prediction of each itailian word using spanish langugae model
