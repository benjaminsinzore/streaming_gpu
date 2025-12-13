from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Serve static files (like your .wav) from the current directory
# WARNING: Only do this in dev or with trusted files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_audio_page():
    if not os.path.exists("cloned_voice.wav"):
        return "<h2>❌ Audio file 'cloned_voice.wav' not found on server.</h2>"
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Play Cloned Voice</title>
        <style>
            body {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f5f5f5;
            }
            h1 {
                color: #333;
            }
            audio {
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>🔊 Cloned Voice</h1>
        <audio controls autoplay>
            <source src="/static/cloned_voice.wav" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        <p><em>Note: If you don't hear sound, check browser autoplay policy or click play manually.</em></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)