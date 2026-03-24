# Yüksek Lisans Araştırma Notları: Pnömoni Tespiti İçin Hibrit Yapay Zeka ve LDM Tabanlı Veri Sentezi

Bu belge, göğüs röntgenlerinden (CXR) pnömoni tespiti yapacak benzersiz bir CNN tabanlı hibrit sınıflandırma modeli geliştirmek ve tıbbi görüntü sentezi (text-to-image) ile veri setini zenginleştirmek amacıyla yapılan literatür taramasını ve metodoloji planını içermektedir.

---

## 1. Temel Makale İncelemesi: RoentGen (2024)

* **Amaç:** Tıbbi veri yetersizliğini gidermek için radyoloji raporlarından sentetik göğüs röntgenleri üretmek.
* **Yöntem:** Stable Diffusion v1.4 (U-Net + CLIP) mimarisinin medikal alana ince ayar (fine-tuning) ile uyarlanması.
* **Veri Seti:** MIMIC-CXR (uzun raporlar kırpıldı, dengesiz sınıflar alt-örnekleme ile dengelendi).
* **Kritik Sonuç:** Üretilen sentetik veriler, hastalık sınıflandırma modellerinin eğitiminde gerçek verilerle eşdeğer veya daha yüksek (AUROC) başarı sağlamıştır. Görüntü kalitesi radyologlarca doğrulanmıştır (FID: 3.6).

---

## 2. Literatür Taraması: Güncel Çalışmalar (2023-2026)

Literatür, araştırmanın doğası gereği birbirini besleyen iki ana kategoriye ayrılmaktadır:

### Grup 1: Tıbbi Görüntü Sentezleme Modelleri (Generatif)
*Metin veya klinik etiketleri kullanarak veri yetersizliğini aşmak için sentetik röntgen üreten mimariler.*

| Makale / Yıl | Kullanılan Yöntem | Odak Noktası ve Başarı |
| :--- | :--- | :--- |
| **XReal** (2024) | Kontrol Edilebilir LDM (Segmentasyon maskeleri ile) | Hastalıkların konumu ve boyutu üzerinde milimetrik uzamsal kontrol sağlanmıştır. |
| **Cheff** (2023) | Kademeli (Cascaded) LDM + Süper Çözünürlük | 1 Megapiksel (1024x1024) klinik teşhis kalitesinde, devasa yüksek çözünürlüklü röntgen üretimi. |
| **Multi-Conditional CXR** (2025) | LDM + Çoklu Koşul Ünitesi (Transformer) | Rapor olmadan sadece klinik etiketlerle (stokastik boş atama) yüksek çeşitlilikte üretim. |
| **Progression-Aware** (2025) | Diziye Duyarlı LDM (Otoregresif Transformer) | Hastalığın (örn. akciğer enfeksiyonu) zaman içindeki ilerleyişi ve ardışık değişimi simüle edilmiştir. |
| **Resource-Efficient LDM** (2025)| LoRA Optimizasyonlu LDM | Dev sunuculara gerek kalmadan, tek standart GPU ile verimli model eğitimi. |

### Grup 2: Hastalık Sınıflandırma ve Teşhis Modelleri (Analitik)
*Gerçek veya sentetik görüntüleri özellik çıkarıcılarla (CNN vb.) analiz edip teşhis koyan mimariler.*

| Makale / Yıl | Kullanılan Yöntem | Odak Noktası ve Başarı |
| :--- | :--- | :--- |
| **LDM-based Augmentation** (2025)| LDM + Sınıflandırma Ağları | Pnömoni gibi hastalıklarda sınıf dengesizliği sentetik veriyle çözülmüş, Youden İndeksi artırılmıştır. |
| **HydraViT** (2025) | Hibrit (CNN + Transformer) Kodlayıcı | CNN'in yerel özellik çıkarımı ile Transformer'ın bağlamsal gücü birleştirilmiş, AUC skoru %1.9 artmıştır. |
| **Robust CXR Classification** (2026)| Difüzyon Temizleme + ResNet18 + Random Forest | Görüntüdeki yanıltıcı gürültüler (adversarial noise) difüzyonla temizlenip %99+ güvenilirlik elde edilmiştir. |

---

## 3. Yeni Makale / Tez İçin Metodoloji Taslağı



Pnömoni teşhisi için geliştirilecek hibrit ağın performansını maksimize etmek ve sentetik veri üretimini sağlamak adına aşağıdaki metodoloji izlenmelidir:

### 3.1. Veri Ön İşleme (Preprocessing)
1.  **Metin Temizliği:** CLIP kodlayıcısı için metinler 77 token sınırı dikkate alınarak özetlenmeli veya CheXpert klinik etiketlerine dönüştürülmelidir.
2.  **Sınıf Dengeleme:** "Sağlıklı" veri yığını alt-örnekleme (undersampling) ile azaltılmalı, nadir pnömoni varyantları için LDM ile sentetik veri üretilerek aşırı örnekleme (oversampling) yapılmalıdır.
3.  **Standardizasyon ve Maskeleme:** Görüntüler sabit boyuta (örn. 512x512) getirilmeli; CNN katmanlarının odaklanmasını artırmak için gerekirse akciğer segmentasyon maskeleri oluşturulmalıdır.

### 3.2. Model Mimarisi Seçimi
* **Üretim Katmanı (Veri Sentezi):** LoRA ile donanım optimizasyonu sağlanmış, ControlNet destekli (uzamsal kontrol için) bir Latent Difüzyon Modeli (Stable Diffusion v1.5 / v2 tabanlı).
* **Teşhis Katmanı (Sınıflandırma):** Görüntüden derin özellikleri çıkaran bir CNN mimarisi ile bağlamsal ilişkileri yakalayan benzersiz bir hibrit sınıflandırıcı.



### 3.3. Test ve Değerlendirme Metrikleri
* **Görüntü Kalitesi:** FID (Fréchet Inception Distance) ve MS-SSIM.
* **Semantik Uyum:** Üretilen sentetik görüntülerin CheXpert/ResNet gibi bağımsız ağlar tarafından doğru hastalık etiketiyle tanınıp tanınmadığı.
* **Klinik ve Pratik Fayda (En Önemlisi):** Geliştirilen CNN tabanlı hibrit sınıflandırıcının *sadece gerçek verilerle* eğitimi ile *gerçek + sentetik verilerle* eğitimi arasındaki AUROC başarı farkının ölçülmesi.

---

## 4. Özgünlük (Novelty) Stratejileri

Standart bir "fine-tuning" işlemi akademik bir makale için yeterince özgün kabul edilmemektedir. Çalışmaya eklenebilecek olası özgün katkılar:

1.  **Mimari Yenilik:** Difüzyon üretim sürecine Transformer yerine bellek dostu **Mamba (State Space Models)** mimarisini entegre etmek.
2.  **Çoklu Koşullandırma (Multimodal):** Sadece metin değil; hastanın ateş, yaş, kan değeri gibi Elektronik Sağlık Kayıtlarını (EHR) da işleyen çoklu kodlayıcılar eklemek.
3.  **Matematiksel Kontrol (Anatomy-aware Loss):** Üretim sırasında radyolojik kurallara (örn. kemik sınırları) uymayı zorlayan yeni bir kayıp fonksiyonu tasarlamak.
4.  **Güvenilirlik Modülü (Self-Correction):** Yanlış sentezlenen hastalık belirtilerini (halüsinasyonları) tespit edip düzelten bir anlık geri bildirim mekanizması kurmak.

---

## 5. Önerilen Veri Setleri

Özellik çıkarımı ve sınıflandırma modelinin hedeflerine en uygun veri setleri şunlardır:

1.  **RSNA Pneumonia Detection Challenge:** Pnömoni teşhisi için kritik olan sınır kutularını (bounding boxes) içerir. CNN özellik çıkarıcılarına lokalizasyon öğretmek için idealdir.
2.  **Kaggle Chest X-Ray (Pediatric Pneumonia):** Sınıf dengesizliği barındırması sebebiyle veri çoğaltma (augmentation) stratejilerini test etmek ve hibrit modeli prototiplemek için harikadır.
3.  **MIMIC-CXR:** 377 binden fazla görüntü ve radyolog raporu barındırır. Metinden görüntüye (text-to-image) sentezleme aşamasının eğitilmesi için altın standarttır.
4.  **ChestX-ray14 / CheXpert:** Çoklu hastalık etiketleri ile modele başlangıç seviyesinde anatomi öğretmek (pre-training) için uygundur.

transformer temelli şeyler konulabilir
tcatch cash
stable diffusion için tcahs yani sonuç çok farklı değilse bu adımı atla kullanılabilir
latent spacei düşürerek sağlayabiliriz


swim transformerlar olabilir uwit vae unet arası kısım transformer ile güncellenebilir.
difusion transformer