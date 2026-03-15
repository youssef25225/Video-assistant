from youtube_transcript_api import YouTubeTranscriptApi


def _extract_video_id(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    else:
        raise ValueError("Could not extract video ID. Please use a standard YouTube link.")


def transcribe(url: str) -> str:
    video_id = _extract_video_id(url)
    ytt = YouTubeTranscriptApi()

    transcript_list = ytt.list(video_id)
    transcript = transcript_list.find_generated_transcript(
        [t.language_code for t in transcript_list]
    )
    fetched = transcript.fetch()

    return " ".join([t.text for t in fetched])