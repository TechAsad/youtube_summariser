import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers.pipelines import pipeline
from textblob import TextBlob
import re
import nltk

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import os



from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title='Youtube Video Summarizer', page_icon="▶️")



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check if the API key exists in the environment variables first
openai_api_key = os.getenv("OPENAI_API_KEY")

# If not found in environment variables, use st.secrets
if not openai_api_key:
    openai_api_key = st.secrets["OPENAI_API_KEY"]


llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini" ,temperature=0.1)


# Ensure that necessary NLTK data is downloaded
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')




# Function to summarize text
def summarize_text(text, max_length=1000):
    summarization_pipeline = pipeline("summarization")
    summary = summarization_pipeline(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']



def summarize_vid(text, max_length=1000):
    print("Generating Summary")

    prompt_chat = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

    you are a helpful assistant, provide a comprehensive summary for the youtube video. The transcript has been provided.
    Summarize the main points and their comprehensive explanations from below text, presenting them under appropriate headings. 
    Use various Emoji to symbolize different sections, and format the content as a cohesive paragraph under each heading. 
    Ensure the summary is clear, detailed, and informative, reflecting the executive summary style found in news articles. 
    Avoid using phrases that directly reference 'the script provides' to maintain a direct and objective tone.
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    video trancript: {text} \n\n 
    word limit: {words}
 
    Summary:
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=[ "text", "words"],
    )


    chain_simple = prompt_chat | llm | StrOutputParser()



    #chain = prompt_chat | llm | 
    response = chain_simple.invoke({ "text": text, "words": max_length})
    return response


# Function to extract keywords
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    keywords = [word for word in words if word not in stop_words and len(word) > 1]

    counter = CountVectorizer().fit_transform([' '.join(keywords)])
    vocabulary = CountVectorizer().fit([' '.join(keywords)]).vocabulary_
    top_keywords = sorted(vocabulary, key=vocabulary.get, reverse=True)[:5]

    return top_keywords


def video_topics(text):

    prompt_chat = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

    you are a helpful assistant, provide only five main topics being discussed in this youtube video. The transcript of the video has been provided.
    just give title to eeach topic. do not write anything else.
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    video trancript: {text} \n\n 
 
    Summary:
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=[ "text"],
    )


    chain_simple = prompt_chat | llm | StrOutputParser()



    #chain = prompt_chat | llm | 
    response = chain_simple.invoke({ "text": text})
    return response



# Function to perform topic modeling (LDA)
def topic_modeling(text):
    vectorizer = CountVectorizer(max_df=2, min_df=0.95, stop_words='english')
    tf = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online', random_state=42)
    lda_model.fit(tf)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
    return topics

# Function to extract YouTube video ID from URL
def extract_video_id(url):
    video_id = None
    patterns = [
        r'v=([^&]+)',  # Pattern for URLs with 'v=' parameter
        r'youtu.be/([^?]+)',  # Pattern for shortened URLs
        r'youtube.com/embed/([^?]+)'  # Pattern for embed URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    return video_id


if "summary" not in st.session_state:
    st.session_state["summary"]= []
    st.session_state["topics"]= []
    st.session_state.clear()

# Main Streamlit app
def main():
    st.title("YouTube Video Summarizer")

    # User input for YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:", "")

    # User customization options
    max_summary_length = st.slider("Max Summary Length:", 100, 500, 1000)

    if st.button("Summarize"):
        #try:
            # Extract video ID from URL
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
                return

            # Get transcript of the video
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            print(transcript[:100])
            if not transcript:
                st.error("Transcript not available for this video.")
                return

            video_text = ' '.join([line['text'] for line in transcript])
            print(video_text[:100])
            # Summarize the transcript
            summary = summarize_vid(video_text, max_length=max_summary_length)
            st.session_state.update({"summary": summary})
            print(summary)
            # Extract keywords from the transcript
            #keywords = extract_keywords(video_text)
            #print(keywords)
            # Perform topic modeling
            topics = video_topics(video_text)
            st.session_state.update({"topics": topics})
            print(topics)
            # Perform sentiment analysis
            #sentiment = TextBlob(video_text).sentiment
            #st.session_state.update({"sentiment": sentiment})
            #print(sentiment)
            
            
            st.subheader("Topics:")
           # for idx, topic in enumerate(topics):
            st.write(st.session_state["topics"])
            #st.write(f"Topic {topics}")
            
            # Display summarized text, keywords, topics, and sentiment
            st.subheader("Video Summary:")
            st.write(st.session_state["summary"])
            #st.write(summary)

            #st.subheader("Keywords:")
            #st.write(keywords)

            #st.subheader("Sentiment Analysis:")
            #st.write(f"Polarity: {sentiment.polarity}")
            #st.write(f"Subjectivity: {sentiment.subjectivity}")
            
            output= f" Topics:\n {topics} \n\n Summary:\n {summary}"
            
            def prepare_download(outputs):
                    content = output
                    
                    return content
            
            st.download_button(
                label="Download Generated Texts",
                data=prepare_download(output),
                file_name=f"Saved_{video_url}.txt",
                mime="text/plain"
            )


       # except TranscriptsDisabled:
        #    st.error("Transcripts are disabled for this video.")
        #except NoTranscriptFound:
        #    st.error("No transcript found for this video.")
        #except Exception as e:
        #    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
