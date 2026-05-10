from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import shutil
import os
import torch
import uuid

app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
RESULT_DIR = "static/results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# load model
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="model/best.pt",
    force_reload=False
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    # unique filename
    uid = uuid.uuid4().hex
    input_path = f"{UPLOAD_DIR}/{uid}_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # run inference
    img = Image.open(input_path)
    results = model(img)

    # detection data
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    # save image with bounding boxes
    results.save(save_dir=RESULT_DIR)

    # get latest output image safely
    result_files = os.listdir(RESULT_DIR)
    result_image = result_files[-1] if result_files else None

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "uploaded_image": f"{uid}_{file.filename}",
            "result_image": result_image,
            "detections": detections
        }
    )