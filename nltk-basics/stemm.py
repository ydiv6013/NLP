# stemming
from nltk.stem import PorterStemmer,SnowballStemmer,LancasterStemmer,RegexpStemmer

# Initialize the stemmer
porter = PorterStemmer()
snawball = SnowballStemmer(language='english')
lancaster = LancasterStemmer()
regex = RegexpStemmer('ing$|s$|e$|able$', min=4)

# List of words to stem
words = ["running", "ran", "runner", "easily", "fairly", "happily"]
 
for word in words : 
    porter_word =  porter.stem(word)
    snawball_word = porter.stem(word)
    lan_word = lancaster.stem(word)
    reg_word = regex.stem(word)
    stem = (word,porter_word,snawball_word,lan_word,reg_word)
    print(stem)
    
    
# Lemmatisation

from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# List of words to lemmatize
words = ["running", "ran", "runner", "easily", "fairly", "happily"]

# Apply lemmatization for nouns and verbs
lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in words]  # 'v' indicates verb

print("Original Words:", words)
print("Lemmatized Words:", lemmatized_words)


import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download WordNet data if not already downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

# Example sentences
sentence1 = "I went to the bank to deposit my money."
sentence2 = "The river bank was teeming with fish."

# Word to disambiguate
word = "bank"

# Use the Lesk algorithm to find the sense of 'bank' in each sentence
sense1 = lesk(word_tokenize(sentence1), word)
sense2 = lesk(word_tokenize(sentence2), word)

# Display the senses
print("Sense in sentence1:", sense1, "-", sense1.definition() )
print("Sense in sentence2:", sense2, "-", sense2.definition() )


# Named Entity recognisation(NER)

sentence3 = "Apple is an American company basaed out of California"
token = nltk.word_tokenize(sentence3)
for word in token:
    #print(f'word is {word}')
    pos_tag = nltk.pos_tag([word])
    #print(f'pos tag is {pos_tag}')
    ne_chunk = nltk.ne_chunk(pos_tag)
    #print(f'ne chunk is {ne_chunk}')
    for chunk in  ne_chunk : 
        if hasattr(chunk,'label'):
            print(chunk.label(),''.join(c[0] for c in chunk))


