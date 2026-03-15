# Proje: RoentGen - MIMIC-CXR Üzerinde Stable Diffusion İnce Ayarı (Fine-Tuning)

## 1. Proje Özeti
[cite_start]Bu proje, genel amaçlı bir metinden-görüntüye (text-to-image) modeli olan `CompVis/stable-diffusion-v1-4` modelinin, tıbbi göğüs röntgeni (CXR) görüntüleri ve klinik raporlar kullanılarak tıp alanına uyarlanmasını (domain-adaptation) amaçlamaktadır[cite: 53, 319, 322]. [cite_start]Hedef, makalede anlatılan "RoentGen" modelinin eğitim sürecini birebir replike etmektir[cite: 63, 69].

## 2. Kullanılacak Kütüphaneler ve Temel Bağımlılıklar
Makaledeki orijinal ortamı kurmak için aşağıdaki kütüphane sürümlerini kullanın:
* [cite_start]`diffusers` (Sürüm 0.7.1) [cite: 320]
* [cite_start]`ViLMedic` (Sürüm 1.2.8) [cite: 320]
* [cite_start]**Model Ağırlıkları:** Hugging Face üzerinden `CompVis/stable-diffusion-v1-4`[cite: 319].

## 3. Veri Seti ve Ön İşleme (Pre-processing)
* [cite_start]**Ana Veri Seti:** MIMIC-CXR (Sürüm 2.0.0)[cite: 292, 294].
* [cite_start]**Eğitim Verisi Seçimi:** Sadece p10-p18 klasörlerindeki veriler kullanılacaktır[cite: 296]. [cite_start]Resmi test setinde (p19 klasörü) bulunan hastalar eğitim setine dahil edilmemelidir[cite: 296].
* **Görüntü İşleme:**
    * [cite_start]Sadece PA (Posterior-Anterior) projeksiyonlu görüntüler filtrelenip kullanılacaktır (yaklaşık 38k örnek elde edilmelidir)[cite: 296, 331].
    * [cite_start]Görüntüler eğitim için 512x512 piksel çözünürlüğe boyutlandırılacaktır[cite: 317].
* **Metin İşleme (Radyoloji Raporları):**
    * [cite_start]Raporların tamamı değil, sadece "Impression" (İzlenim) bölümü prompt olarak kullanılacaktır[cite: 295].
    * [cite_start]7 karakterden kısa olan raporlar veri setinden çıkarılacaktır[cite: 295].
    * [cite_start]CLIP tokenizer sınırı olan 77 token'ı aşan raporlar veri setinden çıkarılacaktır (tüm verilerin yaklaşık %14'üne denk gelir)[cite: 295, 300].

## 4. Model Mimarisi ve Eğitim Stratejisi
* [cite_start]**Güvenlik Filtresi:** Tıbbi promptlarda yüksek yanlış pozitif (false-positive) oranına sahip olduğu için Stable Diffusion'ın yerleşik "safety checker" mekanizması devre dışı bırakılacaktır[cite: 301].
* [cite_start]**Dondurulmuş (Frozen) Bileşenler:** VAE (Variational Autoencoder) ağının ağırlıkları eğitim boyunca güncellenmeyecek, dondurulmuş olarak kalacaktır[cite: 334].
* [cite_start]**Eğitilecek (Trainable) Bileşenler:** Hem U-Net mimarisi hem de CLIP ViT-L/14 metin kodlayıcısı (text encoder) birlikte eğitilecektir (joint fine-tuning)[cite: 58, 229, 300, 336].
* [cite_start]**Kayıp Fonksiyonu (Loss Function):** Örneklenen gürültü ile U-Net tarafından tahmin edilen gürültü arasındaki Ortalama Kare Hatası (Mean Squared Error - MSE) kullanılacaktır[cite: 339, 349, 350].

## 5. Hiperparametreler (En İyi Model Konfigürasyonu)
Makalede en yüksek sadakat (fidelity) skorunu veren ayarlar uygulanacaktır:
* [cite_start]**Eğitim Adımı (Training Steps):** Toplam 60.000 adım[cite: 107].
* [cite_start]**Öğrenme Oranı (Learning Rate):** 5e-5[cite: 107, 135].
* [cite_start]**Batch Size:** 256 (Donanım kısıtlamalarına göre gradient accumulation ile bu değere ulaşılmalıdır)[cite: 317].
* [cite_start]**Hassasiyet (Precision):** Bfloat16 (Donanım destekliyorsa bellek optimizasyonu için)[cite: 317].

## 6. Çıkarım (Inference / Görüntü Üretimi) Ayarları
Eğitim bittikten sonra modelden görüntü üretirken (test aşamasında) aşağıdaki konfigürasyon kullanılacaktır:
* [cite_start]**Noise Scheduler:** PNDM (pseudo numerical methods for diffusion models)[cite: 304].
* [cite_start]**Inference Steps:** 75[cite: 304].
* [cite_start]**Classifier-Free Guidance (CFG) Scale:** 4.0[cite: 304].

## 7. Ajan (Agent) İçin Aksiyon Planı
1. Hugging Face `diffusers` kütüphanesini kullanarak belirtilen model mimarisini yükle. VAE'yi dondur.
2. MIMIC-CXR veri setini indirmek için gerekli kod şablonunu oluştur (veri seti izne tabi olduğu için indirme simüle edilebilir veya yerel bir yol belirtilebilir). Belirtilen metin ve görüntü filtreleme işlemlerini yaz.
3. 512x512 görüntü boyutlandırma ve CLIP tokenizer işlemlerini içeren bir PyTorch Dataset/DataLoader sınıfı oluştur.
4. Belirtilen hiperparametreler, gradient accumulation (gerekiyorsa) ve bfloat16 desteği ile eğitim döngüsünü (training loop) yaz.
5. Eğitilmiş model ile inference yapmak için PNDM scheduler ayarlarını içeren test fonksiyonunu yaz.