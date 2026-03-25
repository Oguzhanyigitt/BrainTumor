# 🧠 Yapay Zeka Destekli Beyin Tümörü Analizi (Karar Destek Sistemi)

Bu proje, Sağlık Bilişimi kapsamında geliştirilmiş; Manyetik Rezonans (MR) görüntüleri üzerinden 4 farklı beyin durumunu (Glioma, Meningioma, Pituitary, No Tumor) sınıflandıran derin öğrenme tabanlı bir web uygulamasıdır.

🔗 **Canlı Uygulama Linki:** [https://braintumorapps.streamlit.app/](https://braintumorapps.streamlit.app/)

---

## 📌 Problem Tanımı ve Önemi
Beyin tümörlerinin manuel olarak MR görüntülerinden tespit edilmesi, uzman radyologlar için bile zaman alıcı ve yorucu bir süreçtir. İnsan gözünden kaçabilecek ufak anomaliler veya yorgunluğa bağlı hatalar, teşhis sürecini olumsuz etkileyebilir. Erken ve doğru teşhis, hayatta kalma oranını doğrudan etkiler. 

Bu projenin amacı; %100 otonom bir teşhis koymak **değil**, triyaj (önceliklendirme) süreçlerini hızlandırarak hekimlere ikinci bir görüş (karar destek sistemi) sunmaktır.

---

## 📊 Veri Seti ve Ön İşleme
Bu çalışmada Kaggle'da açık kaynak olarak sunulan **Brain Tumor MRI Dataset** (Masoud Nickparvar) kullanılmıştır.
* **Sınıflar:** Glioma, Meningioma, No Tumor, Pituitary
* **Görsel Dağılımı:** Eğitim için 4480, Doğrulama (Validation) için 1120 ve Test için 1600 MR görüntüsü.
* **Ön İşleme:** Görüntüler model standartlarına uygun olarak `224x224` piksel boyutuna getirilmiş, piksel değerleri `[0, 1]` aralığına normalize edilmiş ve eğitim sırasında aşırı ezberlemeyi (overfitting) önlemek amacıyla rotasyon, kaydırma ve yakınlaştırma gibi Data Augmentation (Veri Çoğaltma) teknikleri uygulanmıştır.

---

## ⚙️ Model Mimarisi ve Eğitim Süreci
Görüntü işleme ve anomali tespiti için Transfer Learning (Transfer Öğrenme) yöntemiyle **MobileNetV2** mimarisi tercih edilmiştir. 
* **Neden MobileNetV2?** Medikal verilerde yüksek doğruluk sağlarken, web uygulamalarında hızlı yanıt verebilecek kadar hafif ve optimize bir modeldir.
* **Hiperparametreler:** Optimizer olarak `Adam` (Learning Rate: 0.0001), Loss Function olarak `categorical_crossentropy` kullanılmıştır.
* **Eğitim:** Model, ezberlemeyi engellemek için `EarlyStopping` (Erken Durdurma) ve sadece en iyi ağırlıkları saklayan `ModelCheckpoint` callbacks metodlarıyla eğitilmiştir.

---

## 🔬 Model Performansı ve Şüpheci Analiz (Klinik Riskler)
Modelin genel test başarısı (Accuracy) **%82** olarak ölçülmüştür. Ancak tıbbi teşhis modellerinde genel doğruluk oranına körü körüne güvenmek büyük bir mantık boşluğudur. Sınıflandırma raporu, Confusion Matrix ve ROC eğrileri şüpheci bir yaklaşımla incelendiğinde şu kritik sonuçlara ulaşılmıştır:

1. **Ölümcül Hata Riski (Glioma - False Negative):** Modelin en zayıf noktası Glioma tümörleridir (Recall: 0.68). Gerçekte hasta olan vakaların yaklaşık %32'si sistem tarafından kaçırılabilmektedir. Sağlık bilişiminde yanlış
