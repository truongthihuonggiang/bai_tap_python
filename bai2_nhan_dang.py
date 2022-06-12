#0. cài đặt và import thư viện
# pip install Pillow
# pip install opencv-python
# pip3 install torch torchvision torchaudio
# nếu có dùng cuda và vga card
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2

#1. luu model resnet 101 de thuc hien nhan dang
resnet101 = torchvision.models.resnet101(pretrained=True)
resnet101.eval()

#2. doc hinh anh bang thu vien PIL
imgname = "imgs/dautay.jpg";
img = Image.open(imgname).convert('RGB')

#3. tien xu ly hinh anh va chuyen ve dang tensor cua pytorch
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
img_processed = preprocess(img)
# chuyển một tensor thành 1 mảng tensor có 1 phần tử, từ đó mới có thể dùng model dự đoán
batch_img_cat_tensor = torch.unsqueeze(img_processed, 0)
# chuyển model sang trạng thái dùng để dự đoán chứ không phải là trainning
resnet101.eval()

#4.  dự đoán kết quả
out = resnet101(batch_img_cat_tensor)
# đọc danh sách các vật thể được dự đoán theo thứ tự chỉ số
with open('classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
# tìm chỉ số và tên mà model dự đoán
_, index = torch.max(out, 1)
find_name = labels[index[0]]

#5. thể hiện hình ảnh bằng thư viên cv
img_cv = cv2.imread(imgname)
cv2.putText(img_cv,find_name, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
cv2.imshow("vi du nhan dang", img_cv)
cv2.waitKey(0)


