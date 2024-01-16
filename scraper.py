import pandas as pd
import numpy as np
from nrclex import NRCLex
from TwitterAPI import TwitterAPI, TwitterOAuth, TwitterRequestError, TwitterConnectionError, TwitterPager

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import nltk
nltk.download('punkt')

consumer_key = ''
consumer_secret = ''
access_token_key = '' #same thing as access_token
access_token_secret = ''
api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret, api_version='2')

# Done  # auth key with Twitter API

# Done  # READ ME: make sure to !pip install nltk
# Done  # make sure to !pip install TwitterAPI

# Done  # READ ME: this is the dataset I created (NOT KAGGLE) that is available on the drive, make sure it is available
# Done  # in your working directory

sentimentdf = pd.read_csv('Musk_extracted_tweets_2022')

def replies(conversation_id):
    pager_firas = TwitterPager(api, 'tweets/search/recent',
                                 {
                                     'query': f'conversation_id:{conversation_id}',
                                     'tweet.fields': 'author_id,conversation_id,created_at,in_reply_to_user_id'
                                 })
    iterator_firas = pager_firas.get_iterator(wait=5)
   
    count = 0
    replies = []
   
    for item in iterator_firas:
        replies.append(item)
        count = count + 1
   
    return replies, count                   # Returns a list (replies) and an integer (count)

class reply_c:
    def __init__(self, replies, count):
        self.replies = replies  # List
        self.count = count      # Integer
   
    def get_count(self):
        return self.count
   
    def get_replies(self):
        return self.replies
   
    def update_replies(self, more):
        for item in more:
            self.replies.append(item)
        self.count = self.count + len(more)

def apply(dataframe):
    for i in range(1):  # range(len(dataframe))
        reply, count = replies(dataframe.iloc[i]['conversation_id'])
        dataframe.loc[i, 'reply_objects'] = reply_c(reply, count)
               
        print('Completed ', i, ' of 3199')
   
    return dataframe

# Done  # READ ME: Running the following line (make sure to uncomment) will begin populating sentimentdf with replies.
# Done  # When it breaks (which will happen due to too many requests), adjust the function 'apply' so the function can
# Done  # run starting on the row where it broke off.


# Replies -> CSV File

sentimentdf_withreplies = apply(sentimentdf)    # Build the data frame.
sentimentdf_withreplies['reply_objects'] = 0    # Fill the reply_objects column with 0s.
sentimentdf_withreplies = apply(sentimentdf)    # Build again, add the reply objects to the data frame.

sentimentdf["text_replies"] = 0     # Create a column to store text replies (list of strings).
sentimentdf["reply_count"] = 0      # Create a column to store reply counts (int).

sentimentdf["created_at"] = 0
sentimentdf["conversation_id"] = 0
sentimentdf["author_id"] = 0
sentimentdf["in_reply_to_user_id"] = 0
sentimentdf["id"] = 0
sentimentdf["edit_history_tweet_ids"] = 0

text_replies = []                   # Create an outside list of lists to store the text replies.
reply_count = []                    # Create an outside list of ints to store the number of replies.

created_at = []                    
conversation_id = []
author_id = []
in_reply_to_user_id = []
id = []
edit_history_tweet_ids = []

for reply_object in sentimentdf_withreplies['reply_objects']:      # For every reply object in data frame,
    if reply_object != 0:                                          # If the reply object is available,
        reply_object_replies_list = reply_object.replies           # Access its 'replies' attribute (list).
        reply_count.append(reply_object.count)                     # Access its 'count' attribute (int). Append its value to number_replies (list of ints).

        replies_list = []                                          # Create a list to store the text replies.
        created_at_list = []
        conversation_id_list = []
        author_id_list = []
        in_reply_to_user_id_list = []
        id_list = []
        edit_history_tweet_ids_list = []

        for reply_dictionary in reply_object_replies_list:         # For every reply (dictionary) located in the 'replies' attribute (list).
            replies_list.append(reply_dictionary["text"])                                       # Access 'text' and append to a list. 
            created_at_list.append(reply_dictionary["created_at"])                              # Access 'created_at' and append to a list.
            conversation_id_list.append(reply_dictionary["conversation_id"])                    # Access 'conversation_id' and append to a list.
            author_id_list.append(reply_dictionary["author_id"])                                # Access 'author_id' and append to a list.
            in_reply_to_user_id_list.append(reply_dictionary["in_reply_to_user_id"])            # Access 'in_reply_to_user_id" and append to a list.
            id_list.append(reply_dictionary["id"])                                              # Access 'id' and append to a list.
            edit_history_tweet_ids_list.append(reply_dictionary["edit_history_tweet_ids"])      # Access 'edit_history_tweet_ids' and append to a list.

        text_replies.append(replies_list)                          # Append those text replies (list) to an outside list (list of lists).
        created_at.append(created_at_list)
        conversation_id.append(conversation_id_list)
        author_id.append(author_id_list)
        in_reply_to_user_id.append(in_reply_to_user_id_list)
        id.append(id_list)
        edit_history_tweet_ids.append(edit_history_tweet_ids_list)

for i in range(1):  # range(len(dataframe))             # For every tweet's replies (list),      
    sentimentdf["text_replies"][i] = text_replies[i]    # Add them to the dataframe under "text_replies".
    sentimentdf["reply_count"][i] = reply_count[i]
    sentimentdf['created_at'][i] = created_at[i]
    sentimentdf['conversation_id'][i] = conversation_id[i]
    sentimentdf['author_id'][i] = author_id[i]
    sentimentdf['in_reply_to_user_id'][i] = in_reply_to_user_id[i]
    sentimentdf['id'][i] = id[i]
    sentimentdf['edit_history_tweet_ids'][i] = edit_history_tweet_ids[i]

sentimentdf_withreplies.to_csv("data.csv")              # Export the data as a CSV file. 


