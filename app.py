import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# --- SAYFA AYARLARI (Kriter 17 ve 18) ---
st.set_page_config(page_title="Yapay Zeka Destekli Beyin Tümörü Analizi", page_icon="🧠", layout="wide")

# Modelin her seferinde baştan yüklenmesini engelleyerek siteyi hızlandırıyoruz
@st.cache_resource
def get_model():
    return load_model('brain_tumor_model.keras')

model = get_model()
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- YAN MENÜ (KULLANILABİLİRLİK VE GEZİNME - Kriter 18) ---
st.sidebar.title("Menü")
sayfa = st.sidebar.radio("Gezinme", [
    "Ana Sayfa (Tahmin)", 
    "Model Analizi ve Grafikler", 
    "Proje Kodları ve Açıklamaları", 
    "Proje Hakkında ve Sonuç"
])

# --- 1. ANA SAYFA: TAHMİN ARAYÜZÜ (Kriter 19 ve OOD Güvenlik Filtresi) ---
if sayfa == "Ana Sayfa (Tahmin)":
    st.title("Yapay Zeka Destekli Beyin Tümörü Analizi")
    st.write("Bu uygulama, MR görüntüleri üzerinden beyin tümörü tespiti yapmak için eğitilmiş bir derin öğrenme modeli (MobileNetV2) kullanır. Lütfen analiz etmek istediğiniz MR görüntüsünü yükleyin.")
    
    uploaded_file = st.file_uploader("Bir MR Görüntüsü Yükleyin (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Yüklenen Görüntü")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Analiz Sonucu")
            with st.spinner('Yapay zeka görüntüyü inceliyor...'):
                img_array_check = np.array(image)
                
                # --- GÜVENLİK FİLTRESİ 1: KATI RENK KONTROLÜ ---
                # "Ortalama" yerine "Maksimum" (np.max) kullanıyoruz. 
                # Resmin tek bir yerinde bile kırmızı/yeşil/mavi belirginse affetme.
                color_max_std = np.max(np.std(img_array_check, axis=2))
                is_colored = color_max_std > 20.0
                
                # --- GÜVENLİK FİLTRESİ 2: MR ARKA PLAN (PARLAKLIK) KONTROLÜ ---
                # MR görüntüleri karanlıktır (genelde siyah arka plan). 
                # Ortalama parlaklık 0 (siyah) ile 255 (beyaz) arasındadır. Beyaz grafikler 200'ün üzerindedir.
                mean_brightness = np.mean(img_array_check)
                is_too_bright = mean_brightness > 160
                
                # Eğer resim renkliyse VEYA çok aydınlık/beyaz arka planlıysa doğrudan reddet!
                if is_colored or is_too_bright:
                    st.error("⚠️ **Sistem Uyarısı:** Yüklediğiniz görsel yapısal veya renk olarak bir Beyin MR görüntüsüne benzemiyor.Lütfen geçerli bir beyin görüntüsü girin.")
                else:
                    # Görüntü MR testini geçti, Ön İşleme (Kriter 6)
                    img = image.resize((224, 224))
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0  # Normalizasyon
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Tahmin
                    predictions = model.predict(img_array)[0]
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class = class_names[predicted_class_idx]
                    confidence = predictions[predicted_class_idx] * 100
                    
                    # --- GÜVENLİK FİLTRESİ 3: GERÇEKÇİ GÜVEN EŞİĞİ ---
                    # Eşiği 65'ten 45'e düşürdük. Böylece Glioma gibi zorlu ama gerçek vakalar yanlışlıkla elenmeyecek.
                    if confidence < 45.0:
                        st.warning("⚠️ **Kararsız Teşhis:** Görüntü MR formatına uygun ancak model özelliklerden emin olamadı (Güven < %45). Lezyon sınırları belirsiz olabilir.")
                        st.write(f"En yüksek eşleşme (%{confidence:.2f}): {predicted_class}")
                    else:
                        # GÖRÜNTÜ GEÇERLİ VE GÜVEN SKORU YETERLİYSE SONUCU GÖSTER
                        if predicted_class == 'No Tumor':
                            st.success(f"**Teşhis:** Sağlıklı Beyin ({predicted_class})")
                        else:
                            st.error(f"**Teşhis:** Tümör Tespit Edildi - {predicted_class}")
                        
                        st.write(f"**Modelin Güven Skoru:** %{confidence:.2f}")
                        
                        st.write("---")
                        st.write("**Tüm Sınıf Olasılıkları:**")
                        for i, class_name in enumerate(class_names):
                            st.progress(float(predictions[i]), text=f"{class_name}: %{predictions[i]*100:.2f}")
# --- 2. MODEL ANALİZİ SAYFASI (Kriter 14, 15, 16 ve Gelişmiş Metrikler) ---
elif sayfa == "Model Analizi ve Grafikler":
    st.title("📊 Model Performansı ve Kritik Değerlendirme")
    
    # ŞIK METRİK KARTLARI (Dashboard)
    st.markdown("### Temel Performans Metrikleri (Test Seti Üzerinde)")
    st.write("Aşağıdaki metrikler, modelin daha önce hiç görmediği 1600 hastalık test verisi üzerinde hesaplanmıştır. (Not: Canlı tahminlerde etiket bilinmediği için metrikler bilimsel standartlar gereği sabit tutulmuştur.)")
    
    # 3 sütunlu 2 satır metrik kartları oluşturuyoruz
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Genel Doğruluk (Accuracy)", value="%82.0")
    m2.metric(label="F1-Score (Makro Ortalama)", value="%82.0")
    m3.metric(label="Precision (Hassasiyet)", value="%83.0")
    
    m4, m5, m6 = st.columns(3)
    m4.metric(label="Recall (Duyarlılık)", value="%82.0")
    m5.metric(label="ROC AUC Skoru", value="%96.2", delta="Çok Yüksek", delta_color="normal")
    m6.metric(label="Cohen's Kappa", value="0.76", delta="Güçlü Uyum", delta_color="normal")
    
    st.markdown("---")

    st.markdown("""
    ### Sadece Rakamlara Neden Güvenmiyoruz?
    Tıbbi teşhis modellerinde "Accuracy" (Genel Doğruluk) veya MAE (Ortalama Mutlak Hata - ki sınıflandırma için kullanımı bilimsel olarak yanlıştır) gibi metrikler tek başına değerlendirildiğinde büyük yanılgılara yol açabilir. Cohen's Kappa değerimiz (0.76) başarının tesadüf olmadığını kanıtlasa da, şüphe ile yaklaşmak gerekir.
    """)
    
    st.markdown("---")
    
    # EĞİTİM HİPERPARAMETRELERİ (Öne Çıkarılmış)
    st.markdown("### ⚙️ Model Eğitim Hiperparametreleri")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.info("**Model Mimarisi**\n\nMobileNetV2 (CNN)")
    h2.info("**Optimizer**\n\nAdam")
    h3.info("**Learning Rate**\n\n0.0001")
    h4.info("**Batch Size**\n\n32")
    h5.info("**Epoch**\n\n20 (EarlyStop)")
    
    st.write("**Kayıp Fonksiyonu (Loss Function):** `Categorical Crossentropy`")
    
    st.markdown("---")

    # CANLI GRAFİKLER (PLOTLY ENTEGRASYONU)
    st.subheader("📈 Canlı Model Eğitim Süreci (Accuracy & Loss)")
    st.write("Aşağıdaki grafikler interaktiftir. Değerleri görmek için çizgilerin üzerine gelebilir veya istediğiniz bir bölgeye yakınlaşabilirsiniz (zoom).")
    
    # Orijinal eğitim verilerin
    epochs = list(range(1, 21))
    acc = [0.5871, 0.7647, 0.8007, 0.8315, 0.8400, 0.8522, 0.8654, 0.8728, 0.8761, 0.8763, 0.8790, 0.8830, 0.8913, 0.8929, 0.8962, 0.9040, 0.8980, 0.9027, 0.9076, 0.9103]
    val_acc = [0.8259, 0.8446, 0.8509, 0.8759, 0.8750, 0.8884, 0.8830, 0.8893, 0.8884, 0.8964, 0.8973, 0.9000, 0.9018, 0.9054, 0.9116, 0.9080, 0.9089, 0.9116, 0.9152, 0.9054]
    loss = [1.0129, 0.6278, 0.5298, 0.4573, 0.4229, 0.4038, 0.3828, 0.3605, 0.3457, 0.3405, 0.3222, 0.3133, 0.2911, 0.2993, 0.2923, 0.2745, 0.2732, 0.2690, 0.2605, 0.2562]
    val_loss = [0.5730, 0.4553, 0.4051, 0.3628, 0.3550, 0.3203, 0.3175, 0.3046, 0.3005, 0.2845, 0.2794, 0.2744, 0.2677, 0.2563, 0.2546, 0.2647, 0.2402, 0.2445, 0.2394, 0.2395]

    # 1. Canlı Accuracy Grafiği
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=acc, mode='lines+markers', name='Eğitim Doğruluğu', line=dict(color='blue', width=3)))
    fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Doğrulama Doğruluğu', line=dict(color='red', width=3)))
    fig_acc.update_layout(title="Eğitim ve Doğrulama Doğruluğu", xaxis_title="Epoch", yaxis_title="Doğruluk (Accuracy)", hovermode="x unified")
    st.plotly_chart(fig_acc, use_container_width=True)

    # 2. Canlı Loss Grafiği
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', name='Eğitim Kaybı', line=dict(color='blue', width=3)))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Doğrulama Kaybı', line=dict(color='red', width=3)))
    fig_loss.update_layout(title="Eğitim ve Doğrulama Kaybı", xaxis_title="Epoch", yaxis_title="Kayıp (Loss)", hovermode="x unified")
    st.plotly_chart(fig_loss, use_container_width=True)

    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧩 Canlı Karmaşıklık Matrisi")
        st.write("Detayları incelemek için hücrelerin üzerine gelebilirsiniz.")
        
        # Orijinal matris değerlerin
        z = [[272, 70, 26, 32], 
             [16, 256, 32, 96], 
             [3, 5, 390, 2], 
             [1, 0, 0, 399]]
        
        fig_cm = px.imshow(z, text_auto=True, 
                           labels=dict(x="Tahmin Edilen Sınıf", y="Gerçek Sınıf", color="Veri Sayısı"),
                           x=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
                           y=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
                           color_continuous_scale='Blues')
        
        fig_cm.update_layout(title_text="Confusion Matrix", title_x=0.5)
        st.plotly_chart(fig_cm, use_container_width=True)
            
    with col2:
        st.subheader("ROC Eğrisi ve AUC")
        try:
            st.image('roc_curve.png', use_container_width=True)
        except:
            st.warning("ROC Eğrisi bulunamadı.")
# --- YENİ EKLENEN SAYFA: PROJE KODLARI VE AÇIKLAMALARI ---
elif sayfa == "Proje Kodları ve Açıklamaları":
    st.title("💻 Proje Kodları ve Mimari Açıklamalar")
    st.write("Bu bölümde, projenin hem arka planında çalışan eğitim algoritması (Data.py) hem de canlı sistemi ayakta tutan web arayüzü (app.py) kodları şüpheci bir mühendislik yaklaşımıyla açıklanmıştır.")

    # --- 1. DATA.PY KISMI ---
    st.markdown("## 1. Model Eğitimi ve Veri Ön İşleme (`Data.py`)")
    st.info("Bu kod blokları, modelin arka planda nasıl eğitildiğini gösterir. Sistemin orijinal eğitim scriptidir.")
    st.markdown("""
    * **Veri Sızıntısı (Data Leakage) Önlemi:** Test setinde karıştırma (`shuffle=False`) kapatılarak metriklerin şeffaf hesaplanması sağlanmıştır.
    * **Kontrollü Eğitim:** Sistemin kendi haline bırakılıp ezber (overfitting) yapmasını engellemek için `EarlyStopping` kullanılmış, Transfer Learning ile MobileNetV2 mimarisi entegre edilmiştir.
    """)
    st.code('''
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

base_dir = r'C:\\Users\\Oğuzhan YİĞİT\\Dropbox\\PC\\Downloads\\archive'
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Testing')

train_datagen = ImageDataGenerator(
    rescale=1./255,           
    rotation_range=15,        
    width_shift_range=0.1,    
    height_shift_range=0.1,   
    zoom_range=0.1,           
    horizontal_flip=True,     
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) 
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('brain_tumor_model.keras', monitor='val_accuracy', save_best_only=True)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)
    ''', language='python')

    st.markdown("---")

    # --- 2. APP.PY KISMI ---
    st.markdown("## 2. Web Arayüzü ve Güvenlik Filtreleri (`app.py`)")
    st.info("Bu kod bloğu, canlı web sitesinin ve 'Dağılım Dışı Veri' (Out-of-Distribution) güvenlik gardiyanlarının nasıl çalıştığını gösterir.")
    st.markdown("""
    * **OOD (Out-of-Distribution) Tespiti:** Sisteme MR olmayan, renkli veya beyaz arka planlı alakasız görüntüler yüklendiğinde modelin saçmalamasını önlemek için katı piksel (renk ve parlaklık) filtreleri yazılmıştır.
    * **Gerçekçi Güven Eşiği:** Modelin emin olamadığı durumlarda (%45 altı güven skoru) yanlış teşhis riskini önlemek için 'Kararsız Teşhis' uyarısı tetiklenir.
    """)
    st.code('''
# --- GÜVENLİK FİLTRESİ (OOD DETECTION) KODLARI ---
img_array_check = np.array(image)

# Filtre 1: Katı Renk Kontrolü (Maksimum sapma)
# Resmin tek bir yerinde bile kırmızı/yeşil/mavi belirginse affetme.
color_max_std = np.max(np.std(img_array_check, axis=2))
is_colored = color_max_std > 20.0

# Filtre 2: MR Arka Plan (Parlaklık) Kontrolü
# Beyaz grafikler ve aydınlık resimleri engeller.
mean_brightness = np.mean(img_array_check)
is_too_bright = mean_brightness > 160

if is_colored or is_too_bright:
    st.error("⚠️ Sistem Uyarısı: Renkli veya beyaz arka planlı çizim tespit edildi. Lütfen geçerli bir siyah-beyaz MR yükleyin.")
else:
    # Görüntü Ön İşleme
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    
    # Model Tahmini
    predictions = model.predict(img_array)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100
    
    # Filtre 3: Gerçekçi Güven Eşiği
    if confidence < 45.0:
        st.warning("⚠️ Kararsız Teşhis: Model özelliklerden emin olamadı. Lezyon sınırları belirsiz olabilir.")
    else:
        # Sonuç Gösterimi
        st.success(f"Teşhis: {predicted_class} (Güven Skoru: %{confidence:.2f})")
    ''', language='python')
# --- 3. PROJE HAKKINDA SAYFASI (Kriter 1, 2, 3, 4, 5, 8, 9, 10, 11, 20) ---
elif sayfa == "Proje Hakkında ve Sonuç":
    st.title("Proje Detayları ve Sonuç")
    
    st.markdown("### Problem Tanımı ve Önemi")
    st.write("Beyin tümörlerinin manuel olarak MR görüntülerinden tespit edilmesi uzman radyologlar için zaman alıcı bir süreçtir. Erken ve doğru teşhis, hayatta kalma oranını doğrudan etkiler. Bu proje, beyin MR görüntüleri üzerinden tümör tespiti ve sınıflandırmasını otonom hale getirerek hekimlere karar destek sistemi (ikinci bir görüş) sunmayı amaçlamaktadır.")
    
    st.markdown("### Veri Seti (Brain Tumor MRI Dataset)")
    st.write("Bu çalışmada Kaggle'dan Masoud Nickparvar tarafından derlenen veri seti kullanılmıştır. Veri seti 4 sınıftan oluşmaktadır: Glioma, Meningioma, No Tumor ve Pituitary. Eğitim aşamasında toplam 4480, test aşamasında 1600 görsel kullanılmış; veriler 224x224 boyutunda normalize edilerek ve Data Augmentation (veri çoğaltma) teknikleri uygulanarak eğitime hazırlanmıştır.")
    
    st.markdown("### Model Mimarisi ve Eğitim Hiperparametreleri")
    st.write("""
    Projede, web tabanlı teşhis sistemlerinde hızlı ve isabetli sonuç vermesi amacıyla Transfer Learning yöntemiyle **MobileNetV2** mimarisi kullanılmıştır. Modelin eğitim parametreleri (hiperparametreler) şu şekilde ayarlanmıştır:
    * **Optimizasyon (Optimizer):** Adam
    * **Öğrenme Oranı (Learning Rate):** 0.0001 (Medikal verilerin hassasiyeti gözetilerek düşük tutulmuştur)
    * **Kayıp Fonksiyonu (Loss Function):** Categorical Crossentropy
    * **Batch Size:** 32
    * **Epoch:** 20 (Aşırı öğrenmeyi önlemek için EarlyStopping kullanılmış ve 19. Epoch'ta en iyi ağırlıklar kaydedilmiştir.)
    """)

    st.markdown("### Sonuç ve Değerlendirme")
    st.info("""
    Geliştirilen model genel başarı (Accuracy) olarak %82 seviyesine ulaşsa da, %100 otonom bir teşhis aracı olarak kullanılamaz. Sağlık bilişimi etiği gereği, modelin özellikle sınırları belirsiz olan Glioma tümörlerini kaçırma (False Negative) riski dikkate alınmalıdır. Bu uygulama, doktorların yerini almak için değil, triyaj süreçlerini hızlandırmak üzere tasarlanmıştır.
    """)
    
    st.markdown("### Kaynakça")
    st.write("- Dataset: [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)")
    st.write("- Model Mimari: Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.")
