import nltk

"""# download nltk punkt package
nltk.download('punkt')
nltk.download('all')"""

text_data = "Motivated and detail-oriented Machine Learning enthusiast with a master’s degree in Intelligent Vision (Artificial Intelligence) and a strong foundation in Python programming. Skilled in developing innovative machine learning and computer vision solutions, with hands-on experience in creating applications that leverage AI to enhance user experiences. Recently, I have expanded my expertise to include web development with Django and Flask, enabling me to build dynamic, AI-driven web applications.While I am currently building my industrial experience, I am passionate about applying my skills in machine learning to solve real-world problems. I am especially interested in entry-level roles where I can contribute my knowledge in Python, machine learning, and web development, while continuously learning from industry experts and staying updated on the latest advancements in AI and ML. My goal is to collaborate in innovative, team-driven environments and deliver impactful, intelligent solutions that elevate user experience and add value to business objectives."

tokenised_data = nltk.word_tokenize(text_data)

print(tokenised_data)

# bigrams
bigrams = list(nltk.bigrams(tokenised_data))

print(bigrams)

# trigrams

trigrams = list(nltk.trigrams(tokenised_data))

print(trigrams)

# part of speech tagging

for token in tokenised_data :
    pos_tag = nltk.pos_tag(tokens=[token])
    print(pos_tag)
    

# regular expresion tokeniser
from nltk.tokenize import RegexpTokenizer

reg_tokeniser = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
tokens = reg_tokeniser.tokenize(text_data)

for token in tokens : 
    print(nltk.pos_tag([token]))
    

# remove stopwords
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
#print(stop_words)

# remove the stop words from tockenised data

filterd_data = [word for word in tokens if word not in stop_words]

print(filterd_data)

