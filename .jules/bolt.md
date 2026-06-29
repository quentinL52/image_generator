
## 2024-06-29 - Avoid unnecessary image encoding/decoding
**Learning:** Decoding an image from bytes into a `PIL.Image` and then re-encoding it into a different format (like PNG) can add ~40ms overhead (for 1024x1024) and consume extra RAM, just to return an image in a Fast API response.
**Action:** Always return the original byte stream directly (with the correct `media_type`) unless image manipulation is strictly required.
