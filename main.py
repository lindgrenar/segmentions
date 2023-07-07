from collections import Counter
import os
import re
import imageio
from pytube import YouTube
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from moviepy.editor import VideoFileClip
from sqlalchemy import create_engine
from torch import cuda
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

def get_video_comments(api_key, video_id):
    """
    Fetches all comments from a given YouTube video using the YouTube API.

    Args:
        api_key (str): The API key to access the YouTube API.
        video_id (str): The ID of the YouTube video.

    Returns:
        list: A list of comments from the YouTube video.
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return comments


def get_subtitles(video_id):
    """
    Fetches the subtitles of a given YouTube video using the YouTube Transcript API.

    Args:
        video_id (str): The ID of the YouTube video.

    Returns:
        dict: A dictionary where the keys are the start times of the subtitles and the values are the subtitle texts.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        subtitles = {float(item['start']): item['text'] for item in transcript}
        return subtitles
    except:
        print("Subtitles not available for this video.")
        return {}


def filter_comments(comments):
    """
    Filters out comments that don't contain timestamps.

    Args:
        comments (list): A list of comments.

    Returns:
        list: A list of tuples, where each tuple contains a timestamp and a comment.
    """
    filtered_comments = []
    for comment in comments:
        match = re.search(r'(\d+:\d+)', comment)
        if match:
            filtered_comments.append((match.group(0), comment))

    return filtered_comments


def label_timeline(filtered_comments, max_overlap=2.5):
    """
    Groups comments that are close in time into the same segment.

    Args:
        filtered_comments (list): A list of tuples, where each tuple contains a timestamp and a comment.
        max_overlap (float, optional): The maximum allowed time difference (in seconds) for comments to be grouped into the same segment. Defaults to 2.5.

    Returns:
        list: A list of dictionaries, where each dictionary represents a segment and contains the start time, end time, and comments of the segment.
    """
    timeline = []
    for timestamp, comment in filtered_comments:
        start_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(':'))))
        found_overlap = False
        for segment in timeline:
            if abs(segment['start_time'] - start_time) <= max_overlap:
                segment['comments'].append(comment)
                segment['start_time'] = min(segment['start_time'], start_time)
                segment['end_time'] = max(segment['end_time'], start_time + 5)
                found_overlap = True
                break

        if not found_overlap:
            timeline.append({
                'start_time': start_time,
                'end_time': start_time + 5,
                'comments': [comment]
            })
            
    return timeline


def download_video(video_id):
    """
    Downloads the highest resolution mp4 version of a given YouTube video.

    Args:
        video_id (str): The ID of the YouTube video.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    youtube = YouTube(video_url)
    video = youtube.streams.filter(file_extension="mp4", progressive=True).order_by("resolution").desc().first()
    video.download(filename="video.mp4")


def save_to_database(timeline):
    """
    Saves the timeline to a SQLite database.

    Args:
        timeline (list): A list of dictionaries, where each dictionary represents a segment and contains the start time, end time, and comments of the segment.
    """
    df = pd.DataFrame([(ts, c) for ts, comments in timeline.items() for c in comments],
                      columns=['timestamp', 'comment'])
    engine = create_engine('sqlite:///timeline.db')
    df.to_sql('timeline', engine, if_exists='replace', index=False)


def load_emotion_classifier(model_name):
    """
    Loads a pre-trained emotion classification model from the transformers library.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        pipeline: A pipeline object that can be used to classify emotions in text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    return pipeline('text-classification', model=model, tokenizer=tokenizer, device=0 if device == 'cuda' else -1)


def analyze_emotion(text, emotion_classifier, max_length=510):
    """
    Uses the emotion classifier to analyze the emotion of a given text.

    Args:
        text (str): The text to be analyzed.
        emotion_classifier (pipeline): The emotion classifier.
        max_length (int, optional): The maximum length of a text chunk. Defaults to 510.

    Returns:
        str: The most common emotion among the text chunks.
    """
    # Split the input text into smaller chunks
    text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    # Analyze the emotion for each chunk
    chunk_emotions = []
    for chunk in text_chunks:
        result = emotion_classifier(chunk)[0]
        dominant_emotion = result['label'].lower()
        chunk_emotions.append(dominant_emotion)

    # Return the most common emotion among the chunks
    return Counter(chunk_emotions).most_common(1)[0][0]


def cut_video_segments(timeline, emotion_classifier, subtitles):
    """
    Cuts the downloaded video into segments according to the timeline, classifies the emotion of each segment based on the comments and subtitles, and saves each segment as a separate video file.

    Args:
        timeline (list): A list of dictionaries, where each dictionary represents a segment and contains the start time, end time, and comments of the segment.
        emotion_classifier (pipeline): The emotion classifier.
        subtitles (dict): A dictionary where the keys are the start times of the subtitles and the values are the subtitle texts.
    """
    video = VideoFileClip("video.mp4")

    for segment_info in timeline:
        start_time = segment_info['start_time']
        end_time = segment_info['end_time']
        segment = video.subclip(start_time, end_time)

        emotions = [analyze_emotion(comment, emotion_classifier) for comment in segment_info['comments']]
        subtitle_emotions = [analyze_emotion(subtitles[ts], emotion_classifier) for ts in subtitles if start_time <= ts < end_time]
        all_emotions = emotions + subtitle_emotions
        
        emotion_counts = Counter(all_emotions)
        top_emotions = [emotion for emotion, count in emotion_counts.most_common(3)]

        timestamp_str = f"{start_time // 60}:{start_time % 60}"
        file_name = f"{timestamp_str.replace(':', '_')}_{'_'.join(top_emotions)}.mp4"
        segment.write_videofile(file_name)


def main():
    """
    The main function that calls all the other functions. It initializes the API key and video ID, gets the video comments, filters them, labels the timeline, loads the emotion classifier, downloads the video, cuts the video into segments, and saves the timeline to a database.
    """
    api_key = 'YOUR_API_KEY'  # Replace with your YouTube API key
    video_id = 'YOUR_VIDEO_ID'  # Replace with the ID of the YouTube video you want to analyze

    comments = get_video_comments(api_key, video_id)
    filtered_comments = filter_comments(comments)
    timeline = label_timeline(filtered_comments)
    subtitles = get_subtitles(video_id)

    emotion_classifier = load_emotion_classifier('cardiffnlp/twitter-roberta-base-emotion')
    download_video(video_id)
    cut_video_segments(timeline, emotion_classifier, subtitles)
    save_to_database(timeline)


if __name__ == '__main__':
    main()
