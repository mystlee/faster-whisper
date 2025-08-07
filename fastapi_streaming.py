from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

from faster_whisper import WhisperModel

model = WhisperModel("base")
app = FastAPI()


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    def token_generator():
        for token in model.transcribe_stream(file.file):
            yield f"data: {token['text']}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

