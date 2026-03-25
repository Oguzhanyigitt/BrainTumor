import streamlit as st
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
sayfa = st.sidebar.radio("Gezinme", ["Ana Sayfa (Tahmin)", "Model Analizi ve Grafikler", "Proje Hakkında ve Sonuç"])

# --- 1. ANA SAYFA: TAHMİN ARAYÜZÜ (Kriter 19) ---
if sayfa == "Ana Sayfa (Tahmin)":
    st.title("🧠 Yapay Zeka Destekli Beyin Tümörü Analizi")
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
                # Görüntü Ön İşleme (Kriter 6)
                img = image.resize((224, 224))
                img_array = img_to_array(img)
                img_array = img_array / 255.0  # Normalizasyon
                img_array = np.expand_dims(img_array, axis=0)
                
                # Tahmin
                predictions = model.predict(img_array)[0]
                predicted_class_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_class_idx]
                confidence = predictions[predicted_class_idx] * 100
                
                if predicted_class == 'No Tumor':
                    st.success(f"**Teşhis:** Sağlıklı Beyin ({predicted_class})")
                else:
                    st.error(f"**Teşhis:** Tümör Tespit Edildi - {predicted_class}")
                
                st.write(f"**Modelin Güven Skoru:** %{confidence:.2f}")
                
                st.write("---")
                st.write("**Tüm Sınıf Olasılıkları:**")
                for i, class_name in enumerate(class_names):
                    st.progress(float(predictions[i]), text=f"{class_name}: %{predictions[i]*100:.2f}")

# --- 2. MODEL ANALİZİ SAYFASI (Kriter 14, 15, 16) ---
elif sayfa == "Model Analizi ve Grafikler":
    st.title("📊 Model Performansı ve Kritik Değerlendirme")
    
    st.markdown("""
    ### Şüpheci Analiz: %82 Genel Doğruluk Bize Neyi Saklıyor?
    Tıbbi teşhis modellerinde "Accuracy" (Genel Doğruluk) metrik olarak tek başına değerlendirildiğinde büyük yanılgılara yol açabilir. Sınıflandırma raporu ve matrisler detaylıca incelendiğinde modelin bazı mantık boşlukları ve klinik riskler taşıdığı tespit edilmiştir:
    
    1. **Kritik Risk (Glioma - False Negative):** Model, Glioma tümörlerini yakalamada zorlanmaktadır (Recall: 0.68). Gerçekte hasta olan vakaların bir kısmı sistem tarafından kaçırılabilmektedir. Sağlık bilişiminde yanlış negatifler en tehlikeli senaryodur.
    2. **Aşırı Hassasiyet (Pituitary - False Positive):** Model, Pituitary vakalarının tamamını (%100) yakalamış olsa da, emin olamadığı diğer tümör tiplerini de "garanti olsun" mantığıyla Pituitary olarak etiketleme eğilimindedir (Precision: 0.75).
    3. **Sağlıklı Ayrımı:** Model, sağlıklı beyin (No Tumor) görüntülerini yüksek bir doğrulukla (%97) diğerlerinden ayırt edebilmektedir.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Karmaşıklık Matrisi (Confusion Matrix)")
        try:
            st.image('confusion_matrix.png', use_container_width=True)
        except:
            st.warning("Lütfen Evaluate.py dosyasını çalıştırarak confusion_matrix.png dosyasını oluşturduğunuzdan emin olun.")
            
    with col2:
        st.subheader("ROC Eğrisi (ROC Curve)")
        try:
            st.image('roc_curve.png', use_container_width=True)
        except:
            st.warning("Lütfen Evaluate.py dosyasını çalıştırarak roc_curve.png dosyasını oluşturduğunuzdan emin olun.")

# --- 3. PROJE HAKKINDA SAYFASI (Kriter 1, 2, 3, 4, 5, 20) ---
elif sayfa == "Proje Hakkında ve Sonuç":
    st.title("Proje Detayları ve Sonuç")
    
    st.markdown("### Problem Tanımı ve Önemi")
    st.write("Beyin tümörlerinin manuel olarak MR görüntülerinden tespit edilmesi uzman radyologlar için zaman alıcı bir süreçtir. Erken ve doğru teşhis, hayatta kalma oranını doğrudan etkiler. Bu proje, beyin MR görüntüleri üzerinden tümör tespiti ve sınıflandırmasını otonom hale getirerek hekimlere karar destek sistemi (ikinci bir görüş) sunmayı amaçlamaktadır.")
    
    st.markdown("### Veri Seti (Brain Tumor MRI Dataset)")
    st.write("Bu çalışmada Kaggle'dan Masoud Nickparvar tarafından derlenen veri seti kullanılmıştır. Veri seti 4 sınıftan oluşmaktadır: Glioma, Meningioma, No Tumor ve Pituitary. Eğitim aşamasında toplam 4480, test aşamasında 1600 görsel kullanılmış; veriler 224x224 boyutunda normalize edilerek eğitilmiştir.")
    
    st.markdown("### Sonuç (Kriter 20)")
    st.info("""
    Geliştirilen MobileNetV2 tabanlı Transfer Learning modeli, tıbbi görüntü sınıflandırma görevinde genel bir başarı elde etmiş olsa da, %100 otonom bir teşhis aracı olarak kullanılamaz. Sağlık bilişimi etiği gereği, modelin özellikle azınlık sınıflardaki (Glioma) kaçırma oranı dikkate alınmalıdır. Bu uygulama, doktorların yerini almak için değil, triyaj süreçlerini hızlandırmak ve teşhis sürecinde doktora referans olmak üzere tasarlanmıştır. Gelecek çalışmalarda veri setindeki sınıf dağılımları klinik gerçekliğe daha uygun hale getirilmeli ve Siyam ağları gibi daha kompleks özellik çıkarım yöntemleri denenmelidir.
    """)
    
    st.markdown("### Kaynakça")
    st.write("- Dataset: [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)")
    st.write("- Model Mimari: Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.")