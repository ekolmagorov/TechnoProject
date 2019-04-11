import re
import pandas as pd
from pymorphy2 import MorphAnalyzer as ma
from bs4 import BeautifulSoup,Doctype
from bs4.element import Comment
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords 
import os
import codecs
import html2text
import pickle


from multiprocessing import Pool

#SPECIFIC_CHARS = "\t|\n|\!|<|>|\[|\]|\@|\#|\$|\%|\^|\
#		\&|\*|\(|\)|\=|\+|\_|\,|\.|\?|\:|\;|\
#		\№|\/|\"|\'|\\|-|«|»|[0-9]|\| "

def remove_specific_chars(string):

	return re.sub('\s\s+',' ',re.sub('[^a-zA-zа-яА-Я]',' ',string))

def tag_visible(element):

	if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:

		return False
	elif isinstance(element,Comment):
		return False

	return True
			
		
class HTMLPage(object):

	def __init__(self,filename):

		self.name = filename

		with codecs.open(filename,'r',encoding='utf-8') as f:

			soup = BeautifulSoup(f.read(),'html.parser')
			#print(soup.get_text())
			self.title = remove_specific_chars(soup.title.text)
			self.meta = ''
			meta = soup.find_all('meta')

			for tag in meta:
				if 'name' in tag.attrs.keys() and tag.attrs['name'].strip().lower() in ['description', 'keywords']: 
				
					if ('content' in tag.attrs):
						self.meta +=' '+ tag.attrs['content']
					elif ('value' in tag.attrs):
						self.meta += ' '+tag.attrs['value']

			self.meta = remove_specific_chars(self.meta)
			
			tmp = soup.findAll(text=True)
			self.text = ''.join(list(filter(tag_visible,tmp)))
			self.text = html2text.html2text(self.text)
			#print(self.text)
			if (self.text != ''):
				self.url,self.text = self.text.split(' ',1)
				self.text = remove_specific_chars(self.text)
			

	def normalize(self,stemmer_type='pymorphy'):
		#print("ok")
		stopWords = set(stopwords.words('russian'))
	
		if stemmer_type.lower() == 'snowball' or stemmer_type.lower() == 'porter':

			if stemmer_type.lower() == 'snowball':
				stemmer = SnowballStemmer("russian")

			elif stemmer_type.lower() == 'porter':
				stemmer = PorterStemmer("russian",PorterStemmer.NLTK_EXTENSIONS)

		
			

			self.text_words = ' '.join([stemmer.stem(word) for word in self.text.split() 
						    		if word not in stopWords])

			self.meta_words = ' '.join([stemmer.stem(word) for word in self.meta.split() 
								if word not in stopWords])

			self.title_words =  ' '.join([stemmer.stem(word) for word in self.title.split() 
								if word not in stopWords])


		elif stemmer_type.lower() == 'pymorphy':

			parser = ma()

			self.text_words = ' '.join([parser.parse(word)[0].normal_form 
						for word in self.text.split() 
						if word not in stopWords])


			self.meta_words = ' '.join([parser.parse(word)[0].normal_form 
						for word in self.meta.split() 
						if word not in stopWords])
			
			self.title_words = ' '.join([parser.parse(word)[0].normal_form
						 for word in self.title.split() 
						 if word not in stopWords])




class CompetitionPage(HTMLPage):
	
	def __init__(self,filename,doc_id,group_id,target=None):
		super().__init__(filename)
		self.doc_id = doc_id
		self.group_id = group_id
		self.target = target


def load_labels(csv_filename):
	
	labels = pd.read_csv(csv_filename)
	id_groups = {}
	id_targets ={}
	for i in range(len(labels)):
		new_doc = labels.iloc[i]
		doc_group = new_doc['group_id']
		doc_id = new_doc['doc_id']
		target = new_doc['target']
		id_groups[doc_id] = doc_group
		id_targets[doc_id] = target

	return id_groups, id_targets


def load_group(group_id,directory,id_groups,id_targets=None):
	
	list_ids = []
	list_pages = []
	for doc_id,g_id in id_groups.items():
		if g_id == group_id:
			list_ids.append(doc_id)

	print(list_ids)

	for doc_id in list_ids:
		
		filename = os.path.join(directory,'{}.dat'.format(doc_id))
		if not isinstance(id_targets,type(None)):
			p = CompetitionPage(filename,doc_id,
				id_groups[doc_id],id_targets[doc_id])
		else:
			p = CompetitionPage(filename,doc_id,
				id_groups[doc_id])
		p.normalize()
		list_pages.append(p)
		
	return list_pages



if __name__ == '__main__':

	id_groups, id_targets = load_labels('./train_groups.csv')
	for i in range(46,51,1):
		print(i)
		with open("./normalized_groups/group_%d.pickle"%(i),"wb") as f:
	
			ng = load_group(i,'./content',id_groups,id_targets)

			print(len(ng[0].text_words))


			pickle.dump(ng,f)

#a =HTMLPage('./content/7130.dat')
