import streamlit as st
from streamlit_option_menu import option_menu

st.title('Social Media Monitoring')

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Linkdin Fake Text", 'Twitter Fake Text','Twitter Sentiment Analysis'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)


if selected == 'Linkdin Fake Text':
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv('last_data.csv')

    sentences = [sentence.lower()
                .replace('br','')
                .replace('<',"") .replace(">","")
                .replace('\\',"")
                .replace('\/',"")
                for sentence in df.Post]

    print(sentences[5:6], f'\n\nLength Of Data {len(sentences)}')

    import pickle

    with open('twitter-embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    # our_sentence = 'All the things are fake, I allow a candidate to'
    our_sentence = st.text_input('Enter Posts', 'Linkdin Posts are best')

    # lets embed our sentence
    my_embedding = model.encode(our_sentence)

    #Compute cosine similarity between my sentence, and each one in the corpus
    cos_sim = util.cos_sim(my_embedding, embeddings['embeddings'])

    # lets go through our array and find our best one!
    # remember, we want the highest value here (highest cosine similiarity)
    winners = []
    for arr in cos_sim:
        for i, each_val in enumerate(arr):
            winners.append([sentences[i],each_val])

    # lets get the top 2 sentences
    final_winners = sorted(winners, key=lambda x: x[1], reverse=True)



    for arr in final_winners[0:2]:
        print(f'\nScore : \n\n  {arr[1]}')
        st.title(f'\nScore : \n\n  {arr[1]}')
        print(f'\nSentence : \n\n {arr[0]}')
        st.title(f'\nSentence : \n\n {arr[0]}')


####   #######    this is twitter fake data set , and searching for fake statement related to it ##########    


# # comparing the two sentences using SBERT and Cosine Similarity

if selected == 'Twitter Fake Text':
    # here's the install command
    # !pip install -U sentence-transformers
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv('training.1600000.processed.noemoticon.csv',names=['A','B','C','D','E','text'],encoding='latin-1')

    df = df[0:100000]

    sentences = [sentence.lower()
                 .replace('br','')
                 .replace('<',"") .replace(">","")
                 .replace('\\',"")
                 .replace('\/',"")
                 for sentence in df.text]

    # print(sentences[5:6], f'\n\nLength Of Data {len(sentences)}')

    # lets embed the corpus
    # embeddings = model.encode(sentences)

    import pickle

    with open('my-embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    our_sentence = st.text_input('Enter Posts', 'Twitter Posts are best')


    my_embedding = model.encode(our_sentence)

    cos_sim = util.cos_sim(my_embedding, embeddings['embeddings'])

    winners = []
    for arr in cos_sim:
        for i, each_val in enumerate(arr):
            winners.append([sentences[i],each_val])

    # lets get the top 2 sentences
    final_winners = sorted(winners, key=lambda x: x[1], reverse=True)



    for arr in final_winners[0:5]:
        print(f'\nScore : \n\n  {arr[1]}')
        st.title(f'\nScore : \n\n  {arr[1]}')
        print(f'\nSentence : \n\n {arr[0]}')
        st.title(f'\nSentence : \n\n {arr[0]}')



# sentiment analysis

if selected == 'Twitter Sentiment Analysis':

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax

    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)




    labels = ['Negative', 'Neutral', 'Positive']

    # tweet = "@MehranShakarami today's cold @ home 😒 https://mehranshakarami.com"
    tweet = st.text_input('Enter Sentiment Stement', 'Enter The Context of System')
    # tweet = 'Great content! subscribed 😉'

    # precprcess tweet
    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)


    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    for i in range(len(scores)):
        
        l = labels[i]
        s = scores[i]
        st.title(l)
        st.title(s)