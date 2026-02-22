from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_video_chunks(video_id: str):
    """Fetches YouTube transcript and splits it into chunked documents."""
    # Fetch transcript
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    return chunks