import os
import subprocess
import argparse

class YOLOTrainer:
    def __init__(self, dataset_dir, model_type='yolov5s', epochs=50, batch_size=16, image_size=640):
        self.dataset_dir = dataset_dir
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.repo_dir = 'yolov5'

    def prepare(self):
        if not os.path.exists(self.repo_dir):
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'], check=True)
        subprocess.run(['pip', 'install', '-r', f'{self.repo_dir}/requirements.txt'], check=True)

    def create_data_yaml(self):
        data_yaml = f"""
train: {self.dataset_dir}/train/images
val: {self.dataset_dir}/val/images

nc: 1
names: ['vehicle']
"""
        os.makedirs('configs', exist_ok=True)
        with open('configs/custom_data.yaml', 'w') as f:
            f.write(data_yaml.strip())

    def train(self):
        command = [
            'python', f'{self.repo_dir}/train.py',
            '--img', str(self.image_size),
            '--batch', str(self.batch_size),
            '--epochs', str(self.epochs),
            '--data', 'configs/custom_data.yaml',
            '--weights', f'{self.model_type}.pt',
            '--name', 'vehicle_detector'
        ]
        subprocess.run(command, check=True)

    def export_to_onnx(self):
        model_path = os.path.join(self.repo_dir, 'runs/train/vehicle_detector/weights/best.pt')
        export_command = [
            'python', f'{self.repo_dir}/export.py',
            '--weights', model_path,
            '--img', str(self.image_size),
            '--batch', '1',
            '--include', 'onnx'
        ]
        subprocess.run(export_command, check=True)

    def run(self):
        self.prepare()
        self.create_data_yaml()
        self.train()
        self.export_to_onnx()
        print("\n✅ Training complete. Check yolov5/runs/train/vehicle_detector/weights/*.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--model', type=str, default='yolov5s', help='YOLO model variant (e.g., yolov5s, yolov5m)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')

    args = parser.parse_args()

    trainer = YOLOTrainer(
        dataset_dir=args.data_dir,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.img_size
    )
    trainer.run()
