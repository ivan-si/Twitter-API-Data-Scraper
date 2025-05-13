# main_script.py

import pandas as pd
import numpy as np
from nrclex import NRCLex
from TwitterAPI import TwitterAPI, TwitterOAuth, TwitterRequestError, TwitterConnectionError, TwitterPager
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import os
import json
import time
import monetdb.sql # Or use sqlalchemy with a monetdb dialect


# Twitter API Credentials
CONSUMER_KEY = os.environ.get('TWITTER_CONSUMER_KEY', 'YOUR_CONSUMER_KEY')
CONSUMER_SECRET = os.environ.get('TWITTER_CONSUMER_SECRET', 'YOUR_CONSUMER_SECRET')
ACCESS_TOKEN_KEY = os.environ.get('TWITTER_ACCESS_TOKEN_KEY', 'YOUR_ACCESS_TOKEN_KEY')
ACCESS_TOKEN_SECRET = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', 'YOUR_ACCESS_TOKEN_SECRET')

# MonetDB Connection Details
MONETDB_HOSTNAME = os.environ.get('MONETDB_HOSTNAME', 'localhost')
MONETDB_PORT = int(os.environ.get('MONETDB_PORT', 50000))
MONETDB_USERNAME = os.environ.get('MONETDB_USERNAME', 'monetdb')
MONETDB_PASSWORD = os.environ.get('MONETDB_PASSWORD', 'monetdb')
MONETDB_DATABASE = os.environ.get('MONETDB_DATABASE', 'twitter_data')

# Initial Data File
INITIAL_TWEETS_CSV = 'Musk_extracted_tweets_2022.csv' # Make sure this file exists

# --- Initialize Clients & NLTK ---
try:
    api = TwitterAPI(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, api_version='2')
    print("Twitter API client initialized successfully.")
except Exception as e:
    print(f"Error initializing Twitter API client: {e}")
    api = None

analyzer = SentimentIntensityAnalyzer()
print("VADER Sentiment Analyzer initialized.")

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    print("'punkt' tokenizer downloaded.")
print("NLTK setup complete.")

# --- Twitter Data Acquisition ---

class ReplyContainer:
    """
    A container for tweet replies and their metadata.
    """
    def __init__(self, replies_list=None, count=0):
        self.replies = replies_list if replies_list is not None else [] # List of reply dictionaries
        self.count = count                                           # Integer count

    def add_reply(self, reply_item):
        self.replies.append(reply_item)
        self.count += 1

    def add_replies(self, reply_items, new_count):
        self.replies.extend(reply_items)
        self.count += new_count

    def get_reply_texts(self):
        return [reply.get('text', '') for reply in self.replies]

    def __repr__(self):
        return f"<ReplyContainer count={self.count}>"


def fetch_tweet_replies(api_client, conversation_id, max_retries=3, initial_wait_time=60):
    """
    Fetches replies for a given conversation_id with rate limit handling.

    Args:
        api_client (TwitterAPI): Initialized TwitterAPI client.
        conversation_id (str): The conversation ID of the parent tweet.
        max_retries (int): Maximum number of retries for rate-limited requests.
        initial_wait_time (int): Initial wait time in seconds for rate limiting.

    Returns:
        tuple: (list of reply dictionaries, count of replies)
               Returns ([], 0) if fetching fails after retries.
    """
    if not api_client:
        print("Twitter API client not available. Skipping reply fetch.")
        return [], 0

    replies_list = []
    retries = 0
    wait_time = initial_wait_time

    query = f'conversation_id:{conversation_id} is:reply -is:retweet' # Exclude retweets, ensure it's a reply
    tweet_fields = 'author_id,conversation_id,created_at,in_reply_to_user_id,public_metrics,referenced_tweets,lang,entities'

    while retries < max_retries:
        try:
            pager = TwitterPager(api_client, 'tweets/search/recent', {
                'query': query,
                'tweet.fields': tweet_fields,
                'max_results': 100 # Max allowed per request for recent search
            })
            
            batch_replies = []
            for item in pager.get_iterator(wait=5): # internal wait for pagination
                if 'text' in item: # Ensure item is a tweet object with text
                    batch_replies.append(item)
                elif 'message' in item and 'status' in item: # Error object from Twitter
                    print(f"Twitter API error for conversation {conversation_id}: {item['message']} (Status: {item['status']})")
                    if item['status'] == 429: # Rate limit
                        raise TwitterRequestError(item['status'], msg=item['message']) # Trigger retry
                    # For other errors, you might want to log and break or continue carefully
                elif 'data' not in item and 'meta' not in item and len(item.keys()) == 1 and 'title' in item : # Handle cases where item might be an error response not caught by TwitterPager
                    print(f"Skipping non-tweet item: {item}")

            replies_list.extend(batch_replies)
            print(f"Fetched {len(batch_replies)} replies in current batch for conversation_id: {conversation_id}. Total so far: {len(replies_list)}")
            return replies_list, len(replies_list) # Successfully fetched

        except TwitterRequestError as e:
            if e.status_code == 429: # Rate limit exceeded
                print(f"Rate limit exceeded for conversation_id {conversation_id}. Retrying in {wait_time} seconds... (Attempt {retries + 1}/{max_retries})")
                time.sleep(wait_time)
                wait_time *= 2 # Exponential backoff
                retries += 1
            else:
                print(f"TwitterRequestError for conversation_id {conversation_id}: {e.status_code} - {e.message}")
                return [], 0 # Non-rate limit error, stop trying for this ID
        except TwitterConnectionError as e:
            print(f"TwitterConnectionError for conversation_id {conversation_id}: {e}. Retrying in {wait_time} seconds... (Attempt {retries + 1}/{max_retries})")
            time.sleep(wait_time)
            wait_time *= 2
            retries += 1
        except Exception as e:
            print(f"An unexpected error occurred while fetching replies for {conversation_id}: {e}")
            return [], 0 # Other critical error

    print(f"Failed to fetch replies for conversation_id {conversation_id} after {max_retries} retries.")
    return [], 0


def enrich_tweets_with_replies(df, api_client, start_index=0, save_interval=50, checkpoint_file="checkpoint.json"):
    """
    Iterates through a DataFrame of tweets, fetches their replies,
    and stores them in a new column. Handles resuming from a checkpoint.
    """
    if 'reply_objects' not in df.columns:
        # Initialize with None or a more suitable placeholder if not using the custom class immediately
        df['reply_objects'] = pd.Series(dtype='object')


    # Load checkpoint if exists
    processed_conversation_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            try:
                processed_conversation_ids = set(json.load(f))
                print(f"Loaded {len(processed_conversation_ids)} processed conversation IDs from checkpoint.")
            except json.JSONDecodeError:
                print("Could not decode checkpoint file. Starting fresh.")


    total_tweets = len(df)
    for i in range(start_index, total_tweets):
        row = df.iloc[i]
        conversation_id = str(row.get('conversation_id', row.get('id'))) # Use 'id' if 'conversation_id' is missing for the original tweet

        if not conversation_id or pd.isna(conversation_id):
            print(f"Skipping row {i+1}/{total_tweets} due to missing conversation_id.")
            df.at[df.index[i], 'reply_objects'] = ReplyContainer([], 0) # Store empty container
            continue

        if conversation_id in processed_conversation_ids:
            print(f"Skipping already processed conversation_id {conversation_id} (Row {i+1}/{total_tweets}).")
            continue

        print(f"Processing row {i+1}/{total_tweets} - conversation_id: {conversation_id}")
        
        # Check if replies have already been fetched in a previous partial run (if 'reply_objects' already has data)
        current_reply_obj = df.at[df.index[i], 'reply_objects']
        if isinstance(current_reply_obj, ReplyContainer) and current_reply_obj.count > 0:
            print(f"Replies already exist for conversation_id {conversation_id}. Skipping fetch.")
        else:
            replies_data, reply_count = fetch_tweet_replies(api_client, conversation_id)
            df.at[df.index[i], 'reply_objects'] = ReplyContainer(replies_data, reply_count)
        
        processed_conversation_ids.add(conversation_id)

        if (i + 1) % save_interval == 0:
            print(f"Processed {i+1} tweets. Saving checkpoint...")
            with open(checkpoint_file, 'w') as f:
                json.dump(list(processed_conversation_ids), f)
            # Optionally save the DataFrame to CSV/pickle here as well
            # df.to_pickle("tweets_with_replies_checkpoint.pkl")

    print("Finished fetching all replies.")
    # Final checkpoint save
    with open(checkpoint_file, 'w') as f:
        json.dump(list(processed_conversation_ids), f)
    
    return df

# --- Data Processing and Enrichment ---

def analyze_sentiment_vader(text):
    """Analyzes sentiment using VADER and returns the compound score."""
    if not isinstance(text, str) or not text.strip():
        return None  # Return None or np.nan for empty/invalid text
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def analyze_emotions_nrc(text):
    """Analyzes emotions using NRCLex and returns a dictionary of emotions."""
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        emotion = NRCLex(text)
        return emotion.affect_frequencies
    except Exception as e:
        # print(f"Error processing text for NRCLEX: '{text[:50]}...' - {e}") # Commented out to reduce noise
        return {}


def process_and_enrich_data(df):
    """
    Applies sentiment and emotion analysis to original tweets and their replies.
    Flattens reply data for easier storage in a relational database.
    """
    # --- Process Original Tweets ---
    print("Processing original tweets...")
    df['vader_sentiment'] = df['text'].apply(analyze_sentiment_vader)
    df['nrc_emotions'] = df['text'].apply(analyze_emotions_nrc)
    print("Finished processing original tweets.")

    # --- Process Replies ---
    print("Processing replies...")
    all_replies_data = [] # This will store flattened reply data

    for index, row in df.iterrows():
        reply_object = row['reply_objects']
        original_tweet_id = str(row.get('id', 'N/A')) # ID of the original tweet
        original_conv_id = str(row.get('conversation_id', original_tweet_id)) # Conversation ID

        if isinstance(reply_object, ReplyContainer) and reply_object.replies:
            for reply_dict in reply_object.replies:
                reply_text = reply_dict.get('text')
                
                processed_reply = {
                    'reply_id': str(reply_dict.get('id')),
                    'original_tweet_id': original_tweet_id, # Link back to your original tweet's ID
                    'conversation_id': str(reply_dict.get('conversation_id', original_conv_id)),
                    'text': reply_text,
                    'author_id': str(reply_dict.get('author_id')),
                    'created_at': pd.to_datetime(reply_dict.get('created_at')),
                    'in_reply_to_user_id': str(reply_dict.get('in_reply_to_user_id')),
                    'lang': reply_dict.get('lang'),
                    'public_metrics_reply_count': reply_dict.get('public_metrics', {}).get('reply_count', 0),
                    'public_metrics_retweet_count': reply_dict.get('public_metrics', {}).get('retweet_count', 0),
                    'public_metrics_like_count': reply_dict.get('public_metrics', {}).get('like_count', 0),
                    'public_metrics_quote_count': reply_dict.get('public_metrics', {}).get('quote_count', 0),
                    'vader_sentiment': analyze_sentiment_vader(reply_text),
                    'nrc_emotions': json.dumps(analyze_emotions_nrc(reply_text)) # Store as JSON string
                }
                all_replies_data.append(processed_reply)
        # else:
            # print(f"No replies or invalid reply object for original tweet ID {original_tweet_id}")

    replies_df = pd.DataFrame(all_replies_data)
    print(f"Finished processing replies. Extracted {len(replies_df)} replies into a new DataFrame.")
    
    return df, replies_df

# --- MonetDB Interaction ---

def get_monetdb_connection():
    """Establishes a connection to MonetDB."""
    try:
        conn = monetdb.sql.connect(
            hostname=MONETDB_HOSTNAME,
            port=MONETDB_PORT,
            username=MONETDB_USERNAME,
            password=MONETDB_PASSWORD,
            database=MONETDB_DATABASE
        )
        print(f"Successfully connected to MonetDB database '{MONETDB_DATABASE}'.")
        return conn
    except Exception as e:
        print(f"Error connecting to MonetDB: {e}")
        return None

def create_tables(conn):
    """Creates the necessary tables in MonetDB if they don't exist."""
    if not conn:
        return

    cursor = conn.cursor()
    try:
        # Original Tweets Table
        # Adjust column names and types based on your actual initial CSV and needs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS original_tweets (
            id VARCHAR(255) PRIMARY KEY,
            conversation_id VARCHAR(255),
            text TEXT,
            created_at TIMESTAMP,
            author_id VARCHAR(255),
            source VARCHAR(255),          -- Example: from your original CSV
            lang VARCHAR(10),             -- Example: from your original CSV
            -- Add other columns from your 'Musk_extracted_tweets_2022.csv'
            -- For example:
            -- "Unnamed: 0" INTEGER,
            -- "possibly_sensitive" BOOLEAN,
            -- "reply_settings" VARCHAR(255),
            -- "referenced_tweets_type" VARCHAR(50),
            -- "referenced_tweets_id" VARCHAR(255),
            -- "entities_annotations_..." various columns
            -- "public_metrics_retweet_count" INTEGER,
            -- "public_metrics_reply_count" INTEGER,
            -- "public_metrics_like_count" INTEGER,
            -- "public_metrics_quote_count" INTEGER,
            -- "edit_history_tweet_ids" TEXT, -- Store as comma-separated or JSON string
            
            vader_sentiment DOUBLE,
            nrc_emotions TEXT             -- Store as JSON string
        )
        """)
        print("Table 'original_tweets' checked/created.")

        # Tweet Replies Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tweet_replies (
            reply_id VARCHAR(255) PRIMARY KEY,
            original_tweet_id VARCHAR(255), -- FK to original_tweets.id
            conversation_id VARCHAR(255),
            text TEXT,
            author_id VARCHAR(255),
            created_at TIMESTAMP,
            in_reply_to_user_id VARCHAR(255),
            lang VARCHAR(10),
            public_metrics_reply_count INTEGER,
            public_metrics_retweet_count INTEGER,
            public_metrics_like_count INTEGER,
            public_metrics_quote_count INTEGER,
            vader_sentiment DOUBLE,
            nrc_emotions TEXT,                -- Store as JSON string
            FOREIGN KEY (original_tweet_id) REFERENCES original_tweets(id) ON DELETE CASCADE
        )
        """)
        print("Table 'tweet_replies' checked/created.")
        conn.commit()
    except Exception as e:
        print(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        cursor.close()

def clean_value(value, default_for_nan=None):
    """Helper to handle NaN, NaT, and convert to suitable types for DB."""
    if pd.isna(value):
        return default_for_nan
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, list) or isinstance(value, dict): # For NRC emotions or other complex types
        return json.dumps(value)
    return str(value) # Default to string

def insert_original_tweets(conn, df_original_tweets):
    """Inserts data from the original tweets DataFrame into MonetDB."""
    if not conn or df_original_tweets.empty:
        print("No connection or empty DataFrame for original tweets. Skipping insertion.")
        return
    
    cursor = conn.cursor()

    sql = """
    INSERT INTO original_tweets (
        id, conversation_id, text, created_at, author_id, source, lang, 
        vader_sentiment, nrc_emotions 
        -- Add ALL other columns you defined in CREATE TABLE and have in your DataFrame
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) 
    ON CONFLICT (id) DO NOTHING; 
    """ 

    insert_count = 0
    skipped_count = 0

    # Ensure 'id' is string, 'created_at' is datetime for proper conversion
    df_original_tweets['id'] = df_original_tweets['id'].astype(str)
    if 'created_at' in df_original_tweets.columns: # If your CSV has a created_at column
         df_original_tweets['created_at'] = pd.to_datetime(df_original_tweets['created_at'], errors='coerce')
    else: # If not, and it's from Twitter API directly, it should be fine or set to NULL.
        df_original_tweets['created_at'] = pd.NaT


    # Manually list columns to ensure order and existence
    cols_to_insert = [
        'id', 'conversation_id', 'text', 'created_at', 'author_id', 
        'source', 'lang', # These are examples, ensure they are in your df
        'vader_sentiment', 'nrc_emotions'
    ]
    # Prepare a tuple of question marks for the SQL query
    placeholders = ", ".join(["?"] * len(cols_to_insert))
    sql_insert = f"INSERT INTO original_tweets ({', '.join(cols_to_insert)}) VALUES ({placeholders})"


    for _, row in df_original_tweets.iterrows():
        try:
            # Prepare data tuple, ensure all selected columns exist in `row`
            # Handle missing optional columns with None or a default value.
            data_tuple = []
            for col in cols_to_insert:
                if col == 'nrc_emotions': # Already a JSON string from processing
                    data_tuple.append(clean_value(row.get(col), default_for_nan=json.dumps({})))
                elif col == 'created_at':
                     data_tuple.append(row.get(col) if pd.notna(row.get(col)) else None) # MonetDB client handles datetime
                else:
                    data_tuple.append(clean_value(row.get(col)))


            cursor.execute(sql_insert, tuple(data_tuple))
            insert_count += 1
        except Exception as e:
            # This will catch primary key violations if you run it multiple times with same IDs
            # print(f"Error inserting original tweet ID {row.get('id', 'N/A')}: {e}") # Can be noisy
            if "PRIMARY KEY constraint" in str(e) or "unique constraint" in str(e):
                 skipped_count +=1
            else:
                 print(f"Error inserting original tweet ID {row.get('id', 'N/A')}: {e}")
                 conn.rollback() # Rollback on other errors for this row
    
    conn.commit()
    print(f"Inserted {insert_count} original tweets, skipped {skipped_count} (likely duplicates).")
    cursor.close()


def insert_tweet_replies(conn, df_replies):
    """Inserts data from the replies DataFrame into MonetDB."""
    if not conn or df_replies.empty:
        print("No connection or empty DataFrame for replies. Skipping insertion.")
        return

    cursor = conn.cursor()
    
    # Ensure 'created_at' is datetime, handle NaN for numeric, ensure IDs are strings
    df_replies['reply_id'] = df_replies['reply_id'].astype(str)
    df_replies['original_tweet_id'] = df_replies['original_tweet_id'].astype(str)
    df_replies['created_at'] = pd.to_datetime(df_replies['created_at'], errors='coerce')
    
    # Numeric columns that might have NaN from VADER or public_metrics
    numeric_cols = ['vader_sentiment', 'public_metrics_reply_count', 'public_metrics_retweet_count', 
                    'public_metrics_like_count', 'public_metrics_quote_count']
    for col in numeric_cols:
        if col in df_replies.columns:
            df_replies[col] = pd.to_numeric(df_replies[col], errors='coerce')


    # Define the columns for insertion matching the table
    cols_to_insert = [
        'reply_id', 'original_tweet_id', 'conversation_id', 'text', 'author_id',
        'created_at', 'in_reply_to_user_id', 'lang',
        'public_metrics_reply_count', 'public_metrics_retweet_count',
        'public_metrics_like_count', 'public_metrics_quote_count',
        'vader_sentiment', 'nrc_emotions'
    ]
    placeholders = ", ".join(["?"] * len(cols_to_insert))
    sql_insert = f"INSERT INTO tweet_replies ({', '.join(cols_to_insert)}) VALUES ({placeholders})"
    
    insert_count = 0
    skipped_count = 0

    for _, row in df_replies.iterrows():
        try:
            data_tuple = []
            for col in cols_to_insert:
                if col == 'nrc_emotions': # Already json string
                    data_tuple.append(clean_value(row.get(col), default_for_nan=json.dumps({})))
                elif col == 'created_at':
                    data_tuple.append(row.get(col) if pd.notna(row.get(col)) else None)
                elif col in numeric_cols: # Handle potential NaNs for numeric types explicitly as NULL
                    data_tuple.append(clean_value(row.get(col), default_for_nan=None) if pd.notna(row.get(col)) else None)
                else:
                    data_tuple.append(clean_value(row.get(col)))
            
            cursor.execute(sql_insert, tuple(data_tuple))
            insert_count += 1
        except Exception as e:
            if "PRIMARY KEY constraint" in str(e) or "unique constraint" in str(e):
                 skipped_count +=1
            elif "FOREIGN KEY constraint" in str(e):
                print(f"Foreign key constraint violation for reply ID {row.get('reply_id', 'N/A')} (original_tweet_id: {row.get('original_tweet_id', 'N/A')}). This original tweet might not be in 'original_tweets' table. Error: {e}")
                # This can happen if original_tweets insertion failed or was skipped for that ID.
            else:
                 print(f"Error inserting reply ID {row.get('reply_id', 'N/A')}: {e}")
                 # conn.rollback() # Might be too aggressive to rollback all for one row error here
    
    conn.commit()
    print(f"Inserted {insert_count} replies, skipped {skipped_count} (likely duplicates).")
    cursor.close()

def main():
    print("Starting Twitter data processing and ingestion pipeline...")

    # --- 1. Load Initial Data ---
    try:
        original_df = pd.read_csv(INITIAL_TWEETS_CSV)
        # Minimal check for essential columns
        required_cols = ['id', 'conversation_id', 'text'] # 'id' of the tweet, 'conversation_id' for thread
        if not all(col in original_df.columns for col in required_cols):
            print(f"Error: Initial CSV must contain columns: {required_cols}")
            # If 'conversation_id' is not always present for original tweets, it might be the same as 'id'
            # Handle this logic in enrich_tweets_with_replies if 'conversation_id' can be missing
            if 'id' in original_df.columns and 'conversation_id' not in original_df.columns:
                print("Attempting to use 'id' as 'conversation_id' where missing.")
                original_df['conversation_id'] = original_df['id']

            # Re-check after potential fix
            if not all(col in original_df.columns for col in required_cols):
                 print(f"Critical columns {required_cols} still missing. Exiting.")
                 return

        print(f"Loaded initial {len(original_df)} tweets from '{INITIAL_TWEETS_CSV}'.")
    except FileNotFoundError:
        print(f"Error: Initial data file '{INITIAL_TWEETS_CSV}' not found. Exiting.")
        return
    except Exception as e:
        print(f"Error loading initial CSV: {e}. Exiting.")
        return

    # --- 2. Fetch Replies ---
    df_to_process = original_df.copy() # Process all tweets

    print(f"Starting to enrich {len(df_to_process)} tweets with replies...")
    df_with_reply_objects = enrich_tweets_with_replies(df_to_process, api, start_index=0, save_interval=10)
    
    # --- 3. Process and Enrich Data (Sentiment, Emotions, Flatten Replies) ---
    print("Starting data processing and enrichment (sentiment, emotions)...")
    enriched_original_df, flattened_replies_df = process_and_enrich_data(df_with_reply_objects)

    # --- 4. Database Operations ---
    monet_conn = get_monetdb_connection()
    if monet_conn:
        try:
            create_tables(monet_conn)

            # Insert original tweets (now enriched)
            print(f"Attempting to insert {len(enriched_original_df)} enriched original tweets into MonetDB...")
            if 'source' not in enriched_original_df.columns: enriched_original_df['source'] = 'Twitter_CSV_Import'
            if 'lang' not in enriched_original_df.columns: enriched_original_df['lang'] = None # Or detect if possible

            insert_original_tweets(monet_conn, enriched_original_df)
            
            # Insert processed replies
            if not flattened_replies_df.empty:
                print(f"Attempting to insert {len(flattened_replies_df)} processed replies into MonetDB...")
                insert_tweet_replies(monet_conn, flattened_replies_df)
            else:
                print("No replies to insert into MonetDB.")

        except Exception as e:
            print(f"An error occurred during database operations: {e}")
        finally:
            monet_conn.close()
            print("MonetDB connection closed.")
    else:
        print("Could not establish MonetDB connection. Skipping database operations.")

    # --- 5. Save processed data to CSV for inspection ---
    try:
        enriched_original_df.to_csv("enriched_original_tweets_with_analysis.csv", index=False)
        if not flattened_replies_df.empty:
            flattened_replies_df.to_csv("processed_tweet_replies.csv", index=False)
        print("Processed data saved to CSV files locally.")
    except Exception as e:
        print(f"Error saving processed data to CSV: {e}")

    print("Pipeline execution finished.")

if __name__ == '__main__':
    # Ensure you have set your environment variables for API keys and DB credentials
    # e.g.
    # export TWITTER_CONSUMER_KEY="your_key"
    # export TWITTER_CONSUMER_SECRET="your_secret"
    # ... and so on for DB details.
    
    if not all([CONSUMER_KEY != 'YOUR_CONSUMER_KEY', CONSUMER_SECRET != 'YOUR_CONSUMER_SECRET', 
                ACCESS_TOKEN_KEY != 'YOUR_ACCESS_TOKEN_KEY', ACCESS_TOKEN_SECRET != 'YOUR_ACCESS_TOKEN_SECRET']):
        print("Warning: Twitter API credentials are using default placeholders. Please set them via environment variables.")

    main()
