"""
One-time script: Upload all character avatar images to Supabase Storage.

Uploads to bucket 'images_generated' under folder 'character_avatars/'.
Prints the public URL mapping for each character.
"""
import os
import mimetypes
from supabase import create_client

# Supabase credentials (from edgaras_AI .env)
SUPABASE_URL = "https://dwehaynbuioxuzvdaify.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR3ZWhheW5idWlveHV6dmRhaWZ5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTQ2MDU1MCwiZXhwIjoyMDc3MDM2NTUwfQ.X9BILAoY0cAN7l_AAkToeHUusI1O1BkUjSjjDPKgHks"
BUCKET = "images_generated"
FOLDER = "character_avatars"

# Character ID -> filename mapping (matches character_profiles.py)
AVATAR_FILES = {
    "luna": "Luna.jpeg",
    "marco": "Marco.jpg",
    "sofia": "Sofia.jpeg",
    "aria": "Aria.jpg",
    "kai": "Kai.jpeg",
    "elena": "Elena.jpg",
    "jade": "Jade.jpg",
    "mia": "Mia.jpg",
    "nina": "Nina.jpg",
    "lena": "Lena.JPG",
    "zara": "Zara.JPG",
    "rosa": "Rosa.JPG",
    "justin": "Justin.JPG",
    "ivy": "Ivy.JPG",
    "cleo": "Cleo.JPG",
}

CHARACTER_DIR = "/workspace/frontend_demo/character"


def main():
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    results = {}

    for char_id, filename in AVATAR_FILES.items():
        filepath = os.path.join(CHARACTER_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP {char_id}: file not found ({filepath})")
            continue

        # Determine content type
        content_type, _ = mimetypes.guess_type(filepath)
        if not content_type:
            content_type = "image/jpeg"

        # Normalize extension for storage path
        ext = os.path.splitext(filename)[1].lower()
        storage_path = f"{FOLDER}/{char_id}{ext}"

        with open(filepath, "rb") as f:
            image_bytes = f.read()

        print(f"  Uploading {char_id} ({filename}, {len(image_bytes)} bytes) -> {storage_path}")

        try:
            client.storage.from_(BUCKET).upload(
                path=storage_path,
                file=image_bytes,
                file_options={
                    "content-type": content_type,
                    "upsert": "true",
                },
            )
        except Exception as e:
            print(f"  ERROR {char_id}: {e}")
            continue

        public_url = client.storage.from_(BUCKET).get_public_url(storage_path)
        results[char_id] = public_url
        print(f"  OK    {char_id}: {public_url}")

    print("\n--- URL Mapping ---")
    for char_id, url in results.items():
        print(f'    "{char_id}": "{url}",')


if __name__ == "__main__":
    main()
