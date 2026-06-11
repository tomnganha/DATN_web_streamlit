import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ====================== DOUBLE CONV & U-NET ======================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
   
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
       
        # ===== Encoder =====
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
       
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
       
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
       
        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
       
        # ===== Bottleneck =====
        self.bottleneck = DoubleConv(256, 512)
        self.dropout = nn.Dropout2d(0.1)
       
        # ===== Decoder =====
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(512, 256)
       
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
       
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
       
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
       
        # ===== Output =====
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
       
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        p1 = self.pool1(x1)
       
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
       
        x3 = self.down3(p2)
        p3 = self.pool3(x3)
       
        x4 = self.down4(p3)
        p4 = self.pool4(x4)
       
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        bottleneck = self.dropout(bottleneck)
       
        # Decoder
        u4 = self.up4(bottleneck)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.conv4(u4)
       
        u3 = self.up3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.conv3(u3)
       
        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv2(u2)
       
        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv1(u1)
       
        out = self.final_conv(u1)
        return out

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        score = output[0, target_class]
        score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), target_class, output[0].softmax(0)[target_class].item()
    
# ====================== CONFIG ======================
st.set_page_config(page_title="AI Chẩn Đoán Bệnh Da Liễu", layout="wide")

# ====================== CUSTOM CSS ======================
st.markdown("""
    <style>
        img {
            border-radius: 0px !important;
        }
        .stImage img {
            border-radius: 0px !important;
        }
        div[data-testid="stImage"] img {
            border-radius: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ====================== LABEL ======================
labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
label_map = {
    'akiec': 'Tổn thương tiền ung thư do ánh nắng (Actinic Keratoses)',
    'bcc': 'Ung thư tế bào đáy (Basal Cell Carcinoma)',
    'bkl': 'Tổn thương sừng lành tính (Benign Keratosis)',
    'df': 'Dermatofibroma',
    'nv': 'Nốt ruồi melanocytic (Melanocytic Nevi)',
    'vasc': 'Tổn thương mạch máu (Vascular Lesions)',
    'mel': 'Melanoma'
}

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 7)
    )

    model.load_state_dict(torch.load("best_model_mobileNetV2.pth", map_location="cpu"))
    model.eval()
    return model

# ====================== LOAD U-NET ======================
@st.cache_resource
def load_unet_model():
    model = UNet(in_channels=3, out_channels=1)
    
    checkpoint_path = "best_model_Unet.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

# Load model
unet_model = load_unet_model()

def segment_and_crop(image, unet_model, device="cpu"):
    orig_width, orig_height = image.size
    target_size = 256
    
    ratio = min(target_size / orig_width, target_size / orig_height)
    new_width = int(orig_width * ratio)
    new_height = int(orig_height * ratio)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    padded_image = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    padded_image.paste(resized_image, (left, top))
    
    transform_seg = transforms.Compose([transforms.ToTensor()])
    
    input_tensor = transform_seg(padded_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_mask = unet_model(input_tensor)
        pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
    
    pred_mask = cv2.resize(pred_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    binary_mask = binary_mask[top:top+new_height, left:left+new_width]
    binary_mask_resized = cv2.resize(binary_mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    
    # Làm mịn mask
    kernel = np.ones((5,5), np.uint8)
    binary_mask_resized = cv2.morphologyEx(binary_mask_resized, cv2.MORPH_CLOSE, kernel)
    binary_mask_resized = cv2.morphologyEx(binary_mask_resized, cv2.MORPH_OPEN, kernel)
    
    mask_display = (binary_mask_resized * 255).astype(np.uint8)
    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2RGB)
    
    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, mask_display, image
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.15)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(orig_width, x + w + pad_w)
    y2 = min(orig_height, y + h + pad_h)
    
    cropped_image = image.crop((x1, y1, x2, y2))
    
    return image, mask_display, cropped_image

def show_gradcam_streamlit(image, model, grad_cam, transform, device):
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

    orig_width, orig_height = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam, pred_idx, confidence = grad_cam.generate_cam(input_tensor)

    cam = cv2.resize(cam, (orig_width, orig_height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = np.array(image)
    superimposed = heatmap * 0.4 + original * 0.6
    superimposed = np.uint8(superimposed)

    short_label = class_names[pred_idx]
    full_name = label_map[short_label]

    st.subheader("🔍 Giải Thích Mô Hình (Grad-CAM)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="📷 Ảnh Gốc", use_container_width=True)

    with col2:
        st.image(heatmap, caption="🔥 Heatmap", use_container_width=True)

    with col3:
        st.image(superimposed, caption="🎯 Vùng Tập Trung", use_container_width=True)

    st.success(f"{full_name} ({confidence*100:.2f}%)")

# Load models
model = load_model()
target_layer = model.features[18]
grad_cam = GradCAM(model, target_layer)

# Transform
norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# ====================== GIAO DIỆN ======================
st.title("🧠 AI Chẩn Đoán Bệnh Da Liễu")
st.write("Tải lên hình ảnh để phát hiện bệnh da")

file = st.file_uploader("📤 Tải ảnh lên", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="📷 Ảnh đã tải lên", use_container_width=True)

    if st.button("🔍 Dự Đoán"):
        with st.spinner("Đang phân đoạn tổn thương bằng U-Net..."):
            original, mask_colored, cropped_image = segment_and_crop(image, unet_model, "cpu")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(original, caption="📷 Ảnh Gốc", use_container_width=True)
            
            with col2:
                st.image(mask_colored, caption="🟢 Mask Phân Đoạn (U-Net)", use_container_width=True)
            
            with col3:
                st.image(cropped_image, caption="✂️ Ảnh Đã Cắt", use_container_width=True)
            
            with st.spinner("Đang phân loại bệnh..."):
                input_tensor = transform(cropped_image).unsqueeze(0)
                
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).detach().numpy()[0]
                top_idx = np.argsort(probs)[::-1]
                
                pred_label = labels[top_idx[0]]
                confidence = probs[top_idx[0]]

                st.subheader("📊 Kết Quả Dự Đoán")
                st.success(f"**{label_map[pred_label]}** ({confidence*100:.2f}%)")

                st.write("**Top 3 Dự Đoán:**")
                for i in range(3):
                    short = labels[top_idx[i]]
                    st.write(f"{label_map[short]}: **{probs[top_idx[i]]*100:.2f}%**")

                # Mức độ rủi ro
                if pred_label == "mel":
                    st.error("🔴 **Rủi ro cao** - Vui lòng đến bác sĩ ngay lập tức")
                elif pred_label in ["bcc", "akiec"]:
                    st.warning("🟡 **Rủi ro trung bình** - Nên theo dõi và khám sớm")
                else:
                    st.info("🟢 **Rủi ro thấp**")

                # Grad-CAM
                show_gradcam_streamlit(cropped_image, model, grad_cam, transform, "cpu")

                st.warning("⚠️ Đây chỉ là công cụ hỗ trợ tham khảo, không thay thế chẩn đoán của bác sĩ chuyên khoa.")