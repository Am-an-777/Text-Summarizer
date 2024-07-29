import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation  
from heapq import nlargest

text= """ The abstractive approach involves summarization based on deep learning. So, it uses new phrases and terms, 
different from the actual document, keeping the points the same, just like how we actually summarize. 
So, it is much harder than the extractive approach.
It has been observed that extractive summaries sometimes work better than the abstractive ones probably because extractive 
ones don't require natural language generations and semantic representations."""

def summarizer(rawdocs):
    # Convert stop words set to list
    stopwords = list(STOP_WORDS)
    
    # Load the spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Process the document
    doc = nlp(rawdocs)
    
    # Tokenize the document
    tokens = [token.text for token in doc]
    
    # Calculate word frequencies
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1
                
    # Find the maximum frequency
    max_freq = max(word_freq.values())
    
    # Normalize word frequencies
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq
    
    # Tokenize the document into sentences
    sent_tokens = [sent for sent in doc.sents]
    
    # Calculate sentence scores
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]
    
    # Determine the number of sentences to include in the summary
    select_len = int(len(sent_tokens) * 0.3)
    
    # Select the highest scoring sentences
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    
    # Join the selected sentences to form the final summary
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    
    # Return the summary and additional information
    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))
