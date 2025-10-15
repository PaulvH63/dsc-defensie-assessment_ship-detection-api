
# Data Science Cell Assessment - Airbus Ship Detection

### Files included
- dsc_defensie_assessment.ipynb (image & mask visualizations, inference and evaluation)(**Part 1 of assignment**)
- app/main.py (FastAPI app with 2 end points) (**Part 2 of assignment**)
- src/model.py (ShipDetector object for model loading & prediction)
- src/evaluation.py (evaluation helpers)
- src/utils.py (rle encoding and decoding helpers)
- requirements.txt (dependencies)

### Run locally (dev)
1. Create a virtual environment and install dependencies
```sh
python3 -m venv venv-shipdetector
```

2. Activate environment (or select environment in VSCode to run Jupyter Notebook)
```sh
# macOS / Linux
source venv-shipdetector/bin/activate

# Windows
venv-shipdetector\Scripts\activate
```

3. Install dependencies into environment
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the Notebook `dsc_defensie_assessment.ipynb`


5. Start API from within project ROOT:
```sh
docker build -t shipdetector-api:lastest .
docker run --rm -p 8000:8000 -e MODEL_PATH="models/yolo11n-seg.pt" shipdetector-api:latest
```
6. Try it out: upload image and see mask or contour results: http://localhost:8000/docs



### Assumptions
- Used pre-trained YOLO segmentation nano for simplicity
- All resulting masks (all classes) will be considered as boat for simplicity