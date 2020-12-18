# from google.cloud import bigquery

import os
import pandas as pd
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize 
import string
import math 
import re 
import time
import heapq

class DataPreProcessing:

	def __init__(self):
		self.tokens = {}
		self.length={}
		self.df = None

	def tokenize_and_remove_punctuations(self, s):
		translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
		modified_string = s.translate(translator)
		modified_string = ''.join([i for i in modified_string if not i.isdigit()])
		return word_tokenize(modified_string)

	def get_stopwords(self):
		stop_words = [word for word in open('stopwords.txt','r').read().split('\n')]
		return stop_words
	
	def remove_stop_words(self, tokens):
		stop_words = self.get_stopwords()
		filtered_words = [token for token in tokens if token not in stop_words and len(token) > 2]
		return filtered_words

	def stem_words(self, tokens):
		stemmer = PorterStemmer()
		stemmed_words = [stemmer.stem(token) for token in tokens]
		return stemmed_words

	def dataExtraction(self):
		# df = pd.read_csv('./github_issues.csv', names=["issue_url", "issue_title", "body"])
		df = pd.read_csv('./Data.csv')
		self.total_no_of_docs = len(df)

		for i in range(len(df)):
			text = ""
			text += df.loc[i, "issue_url"]
			text += " " + df.loc[i, "issue_title"]
			text += " " + df.loc[i, "body"]

			tokens = self.tokenize_and_remove_punctuations(text)
			filtered_words = self.remove_stop_words(tokens)
			stemmed_words = self.stem_words(filtered_words)

			self.tokens[i+1] = stemmed_words
			self.length[i+1] = len(stemmed_words)
		# print(len(self.length))
			
			
	
class InvertedIndex:

	def __init__(self):
		self.PostingList = {}
		self.tf_idf_scores = {}
		self.avg_tokens_per_doc = 0
		self.total_tokens = 0
		self.total_no_of_docs = 1000000

	def updateTokensData(self):
		self.total_tokens = len(self.PostingList)
		self.avg_tokens_per_doc = self.total_tokens // self.total_no_of_docs

	def printTokensInfo(self):
		print("Total No of docs : {}".format(self.total_no_of_docs))
		print("Total No of tokens : {}".format(self.total_tokens))
		# print("Avg no of tokens per doc: {}".format(self.avg_tokens_per_doc))

	def InvIndex(self, tokens):
		for doc, tokens in tokens.items():
			for word in tokens:
				if word in self.PostingList.keys():
					self.PostingList[word].append(doc)
				else:
					self.PostingList[word] = [doc]

	def TermFrequency(self):
		Index = ""
		for word, lst in self.PostingList.items():
			text = ""
			text = word + "(" + str(len(lst)) + ") : [ "
			cnt = 0
			for i in range(len(lst)):
				doc = lst[i]
				if i>0 and lst[i-1]!=lst[i]:
					text += str(int(lst[i-1])) + "(" + str(cnt) + "), "
					cnt = 1
				else:
					cnt += 1  
			text += str(int(lst[len(lst)-1])) + "(" + str(cnt) + ")"
			text += " ]\n"
			Index += text
			

		f = open('./PostingList.txt', 'w')
		f.write(Index)
		f.close()
		


class Query:

	def __init__(self):
		self.scores = {}
		self.length = {}
		self.terms = {}

	def preProcess(self, query, invIndex, length):
		k = DataPreProcessing()
		tokens = k.tokenize_and_remove_punctuations(query)
		filtered_words = k.remove_stop_words(tokens)
		stemmed_words = k.stem_words(filtered_words)
		self.terms = stemmed_words
		self.Scores(invIndex, tokens)
		
	def Scores(self, invIndex, length):
		for word in self.terms:
			if word in invIndex.PostingList.keys():
				p = invIndex.PostingList[word]
				lst = set(p)
				for i in lst:
					if i in self.scores.keys():
						self.scores[i] += ( math.log10(invIndex.total_no_of_docs/len(lst)) * (1 + math.log10(p.count(i))) ) 
					else:
						self.scores[i] = ( math.log10(invIndex.total_no_of_docs/len(lst)) * (1 + math.log10(p.count(i))) ) 

		# for doc in self.scores.keys():
		# 	print(doc)
		# 	print(length[doc])
		# 	self.scores[doc] = self.scores[doc]/length[doc]
		
		self.printScores()
	
	def printScores(self):
		df = pd.read_csv('./Data.csv', names=["issue_url", "issue_title", "body"])
		df = df.head(100000)
		i=0
		for doc in sorted(self.scores, key=self.scores.get, reverse=True):
			if i<10:
				print(doc, self.scores[doc], df.loc[doc-1, "issue_url"])
				i+=1
			else:
				break
		

				
start = time.time()
k = DataPreProcessing()
k.dataExtraction()

index = InvertedIndex()
index.InvIndex(k.tokens)

Inverted_Index = index.PostingList

index.updateTokensData()
index.printTokensInfo()
index.TermFrequency()


query = Query()  
print("Enter the query: ")
inp = input()
query.preProcess(inp,index,k.length)
end = time.time()
print(end - start)

