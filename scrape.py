import os
import re
import sys

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

API_KEY = os.environ.get("YOUTUBE_API_KEY")

VIDEOS = [
    "https://www.youtube.com/watch?v=T0gHkSh2ExE",
    "https://www.youtube.com/watch?v=akdVN7HTQN8",
    "https://www.youtube.com/watch?v=hHIcYvg1PlA",
    "https://www.youtube.com/watch?v=PRQ9_Hu-bXk",
    "https://www.youtube.com/watch?v=jhKxOzdd_cY",
    "https://www.youtube.com/watch?v=CRU2RoTSFVA",
    "https://www.youtube.com/watch?v=ltm23zexfHk",
    "https://www.youtube.com/watch?v=CwsGTa_QZiM",
    "https://www.youtube.com/shorts/WeQSKsZuhmI",
    "https://www.youtube.com/watch?v=VavXBDH7N0w",

]

# Set to True to also search YouTube for these and collect comments from the results.
USE_SEARCH = False
SEARCH_QUERIES = [
    "fuel price hike Philippines",
    "oil price increase Philippines",
    "transport strike Philippines",
    "jeepney strike",
    "tigil pasada",
    "PUV modernization strike",
]
VIDEOS_PER_QUERY = 10  # each search costs 100 of your 10,000 daily quota units

# Max comments per video (None = all of them). Lower = more variety across videos.
MAX_COMMENTS_PER_VIDEO = 1000
INCLUDE_REPLIES = False

# Output file (saved next to this script).
OUTPUT_FILE = "youtube_comments.csv"

VIDEO_TITLES = {}  # filled in by search: video_id -> title


def extract_video_id(value):
    """Accept a full YouTube URL, a youtu.be link, a Shorts link, or a bare ID."""
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    raise ValueError(f"Could not find a video ID in: {value}")


def search_videos(youtube, query, count):
    """Return {video_id: title} for the top YouTube results for a search query (Philippines region)."""
    response = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        regionCode="PH",
        maxResults=count,
        order="relevance",
    ).execute()
    return {item["id"]["videoId"]: item["snippet"]["title"] for item in response.get("items", [])}


def comment_row(video_id, item, comment_type, parent_id=None):
    s = item["snippet"]
    return {
        "video_id": video_id,
        "video_title": VIDEO_TITLES.get(video_id, ""),
        "comment_id": item["id"],
        "type": comment_type,
        "parent_id": parent_id,
        "author": s.get("authorDisplayName"),
        "text": s.get("textOriginal"),
        "likes": s.get("likeCount", 0),
        "published_at": s.get("publishedAt"),
        "updated_at": s.get("updatedAt"),
    }


def fetch_replies(youtube, video_id, parent_id):
    rows = []
    page_token = None
    while True:
        response = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText",
        ).execute()
        for item in response.get("items", []):
            rows.append(comment_row(video_id, item, "reply", parent_id))
        page_token = response.get("nextPageToken")
        if not page_token:
            return rows


def fetch_comments(youtube, video_id, limit):
    rows = []
    top_level_count = 0
    page_token = None
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText",
            order="time",
        ).execute()

        for thread in response.get("items", []):
            top = thread["snippet"]["topLevelComment"]
            rows.append(comment_row(video_id, top, "comment"))
            top_level_count += 1

            if INCLUDE_REPLIES and thread["snippet"].get("totalReplyCount", 0) > 0:
                rows.extend(fetch_replies(youtube, video_id, top["id"]))

            if limit is not None and top_level_count >= limit:
                return rows

        page_token = response.get("nextPageToken")
        if not page_token:
            return rows


def main():
    if not API_KEY:
        sys.exit("API key not found. Run: setx YOUTUBE_API_KEY \"your key\" then open a new terminal.")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    all_rows = []

    # Collect video IDs: manual list first, then search results (duplicates removed).
    video_ids = []
    for video in VIDEOS:
        try:
            video_ids.append(extract_video_id(video))
        except ValueError as e:
            print(f"Skipping: {e}")

    for query in (SEARCH_QUERIES if USE_SEARCH else []):
        print(f'Searching YouTube for "{query}" ...')
        try:
            found = search_videos(youtube, query, VIDEOS_PER_QUERY)
        except HttpError as e:
            print(f"  Search failed: {e}")
            continue
        VIDEO_TITLES.update(found)
        video_ids.extend(found)

    video_ids = list(dict.fromkeys(video_ids))
    print(f"\n{len(video_ids)} videos to collect from.\n")

    for video_id in video_ids:
        print(f"Collecting comments from {video_id} {VIDEO_TITLES.get(video_id, '')[:60]} ...")
        try:
            rows = fetch_comments(youtube, video_id, MAX_COMMENTS_PER_VIDEO)
        except HttpError as e:
            reason = e.error_details[0].get("reason") if e.error_details else e.resp.status
            print(f"  Failed ({reason}). Comments may be disabled, or the key/quota is invalid.")
            continue

        print(f"  Got {len(rows)} rows.")
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("No comments collected.")

    df = pd.DataFrame(all_rows).drop_duplicates(subset="comment_id")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    # utf-8-sig so Excel shows emojis and non-English text correctly
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} comments to {output_path}")


if __name__ == "__main__":
    main()
