Next Steps on Raspberry Pi:

--------------------
Install ONNX Runtime
--------------------
pip install onnxruntime

----------------------------------------
Place yolov5s.onnx in models/ directory:
----------------------------------------

You can generate it using:

git clone https://github.com/ultralytics/yolov5
cd yolov5

pip install -r requirements.txt

python export.py --weights yolov5s.pt --img 640 --batch 1 --include onnx

------------
Run the app:
------------
python3 main.py
