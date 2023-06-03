import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('GPU')
else:
  device = torch.device('cpu')
  print('CPU')

class ChineseCharDataset(Dataset):
  def __init__(self, data_file, root_dir, dict_file):
    # data_file:  標註檔的路徑 (標註檔內容: ImagePath, GroundTruth)
    # root_dir: ImagePath所在的資料夾路徑
    # dict_file: 字典的路徑

    # 使用 pandas 將生成的單字labels.txt當作csv匯入進來
    self.char_dataframe = pd.read_csv(data_file, index_col=False, encoding='utf-8', header=None)
    self.root_dir = root_dir
    with open(dict_file, 'r', encoding='utf-8') as f:
      # 將資料集包含的字集匯入進來
      word_list = [line for line in f.read().split('\n') if line.strip() != '']
      self.dictionary = {word_list[i]: i for i in range(0, len(word_list))}

    print(self.char_dataframe)
    print(self.dictionary)

  def __len__(self):
    return len(self.char_dataframe)

  def __getitem__(self, idx):
    
    # 取得第idx張圖片的path，並將圖片打開
    image_path = os.path.join(self.root_dir, self.char_dataframe.iloc[idx, 0])
    image = Image.open(image_path)
    
    # 取得 Ground Truth 並轉換成數字
    char = self.char_dataframe.iloc[idx, 1]
    char_num = self.dictionary[char]

    
    return (transforms.ToTensor()(image), torch.Tensor([char_num]))
  
# 宣告好所有要傳入 ChineseCharDataset 的引數
data_file_path = './output/labels.txt'
root_dir = './'
dict_file_path = './chars.txt'

# 模型儲存位置
save_path = './checkpoint.pt'

# 宣告我們自訂的Dataset，把它包到 Dataloader 中以便我們訓練使用
char_dataset = ChineseCharDataset(data_file_path, root_dir, dict_file_path)
char_dataloader = DataLoader(char_dataset, batch_size=64, shuffle=True, num_workers=2)


# 创建一个与原始模型相同结构的模型实例
net = models.resnet18(num_classes=40)

# 加载保存的权重
checkpoint = torch.load(save_path, map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])

# 设置模型为评估模式
net.eval()

# 准备待推理的图像
image_path = './test/img_0000014.jpg'
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0)

# 使用模型进行推理
with torch.no_grad():
    outputs = net(input_image)

# 获取预测结果
_, predicted = torch.max(outputs, 1)
prediction = predicted.item()

# 输出预测结果
print('Prediction:', prediction)


# 定义类别标签映射
class_labels = ['肉','古','幼','酥','成','傢','婦','汎','貨','理','男','大','老','樹','民','鴻','禾','髮','酒','麗','鹽','容','由','寵','中','速','食','汽','子','院','批','洗','素','我','快','雞','出','動','品','活']  # 替换为你的实际类别标签

# 获取预测结果
_, predicted = torch.max(outputs, 1)
prediction = predicted.item()

# 根据类别索引获取类别标签
predicted_label = class_labels[prediction]

image = Image.open(image_path)

plt.imshow(image)
plt.axis('off')

#plt.text("Prediction result:",predicted_label)

# 输出预测结果
print('Prediction word:', predicted_label)
plt.show()