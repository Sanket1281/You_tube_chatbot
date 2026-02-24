from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_video_chunks(video_id: str):
    """Fetches YouTube transcript and builds semantically clean chunks with metadata."""
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

    # 1. Group transcript into larger "Time Blocks" first
    raw_docs = []
    current_text = ""
    current_start_time = 0.0

    for item in transcript_list:
        # ---- THE FIX: Handle the new library update safely ----
        if isinstance(item, dict):
            item_text = item['text']
            item_start = item['start']
        else:
            item_text = item.text
            item_start = item.start

        # Log the start time of the new block
        if not current_text:
            current_start_time = item_start

        current_text += item_text + " "

        # Group into ~2000 character blocks (roughly 2-3 minutes of speaking)
        if len(current_text) >= 2000:
            minutes = int(current_start_time // 60)
            seconds = int(current_start_time % 60)
            timestamp = f"{minutes}:{seconds:02d}"

            doc = Document(
                page_content=current_text.strip(),
                metadata={"timestamp": timestamp, "video_id": video_id}
            )
            raw_docs.append(doc)
            current_text = ""

    # Catch any leftover text at the very end of the video
    if current_text:
        minutes = int(current_start_time // 60)
        seconds = int(current_start_time % 60)
        timestamp = f"{minutes}:{seconds:02d}"

        doc = Document(
            page_content=current_text.strip(),
            metadata={"timestamp": timestamp, "video_id": video_id}
        )
        raw_docs.append(doc)

    # 2. Apply Semantic Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    # Split the blocks (This automatically preserves the timestamp metadata!)
    final_chunks = splitter.split_documents(raw_docs)

    return final_chunks