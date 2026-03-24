# RoentGen: MIMIC-CXR Üzerinde Stable Diffusion İnce Ayarı (Fine-Tuning) - Proje İncelemesi

Bu doküman, projede yer alan dosyaların, mimarilerin ve eğitim/çıkarım süreçlerinin kapsamlı ve detaylı bir incelemesidir. 

## 1. Projenin Amacı ve Genel Bakış
"RoentGen" projesi, genel amaçlı bir metinden-görüntüye (text-to-image) modeli olan `CompVis/stable-diffusion-v1-4` modelini tıp alanına, özellikle **göğüs radyolojisine (Chest X-Ray / CXR)** uyarlamayı hedefler. Temel amaç, gerçekçi radyoloji raporları (özellikle raporun "İzlenim/Impression" bölümü) verildiğinde, bu raporla anlamsal olarak uyumlu, yüksek sadakatli (high-fidelity) sentetik göğüs röntgeni görüntüleri üretebilen bir model eğitmektir. Bu süreç MIMIC-CXR veri seti kullanılarak "domain-adaptation" (alan uyarlaması) yaklaşımı ile gerçekleştirilir.

## 2. Mimari ve Eğitim Stratejisi
Projede Stable Diffusion modelinin bileşenlerine özel bir ince ayar stratejisi uygulanmıştır:
* **Dondurulan (Frozen) Bileşen: VAE (Variational Autoencoder):** Modelin görüntüleri latent (gizli) uzaya sıkıştırdığı ve geri açtığı bileşen. Ağırlıkları güncellenmez.
* **Eğitilen Bileşenler (Joint Fine-tuning):** Hem görüntüyü oluşturan **U-Net** mimarisi hem de metni anlayan **CLIP ViT-L/14 metin kodlayıcısı (Text Encoder)** beraber eğitilir. Bu sayede modelin radyolojik terimleri (örn. "consolidation", "pleural effusion") ve bunların görsel karşılıklarını öğrenmesi hedeflenir.
* **Kayıp Fonksiyonu (Loss):** Modele eklenen gürültü ile U-Net'in tahmin ettiği gürültü arasındaki MSE (Ortalama Kare Hatası) kullanılır.
* **Güvenlik Filtresi (Safety Checker):** Tıbbi terimlerde yanlış pozitif (false-positive) ürettiği için çıkarım aşamasında tamamen devre dışı bırakılır.

## 3. Veri Seti ve Ön İşleme Süreci
Veri işleme, modelin başarısı için kritik bir adımdır ve `dataset.py` dosyası içinde yönetilir. 
* **Veri Seti:** MIMIC-CXR (Sürüm 2.0.0). Resmi test setini içeren `p19` klasöründeki hastalar hariç tutulur (Data Leakage engellemek için).
* **Görüntü Filtreleme:** Sadece PA (Posterior-Anterior) projeksiyonlu göğüs röntgenleri seçilir. Eğitim aşamasında görüntüler standart olarak `512x512` çözünürlüğe boyutlandırılır.
* **Metin Filtreleme:** Raporların sadece **"Impression" (İzlenim)** kısmı kullanılır. 7 karakterden kısa olan veya CLIP tokenizer'ının sınırı olan 77 token'ı geçen raporlar elenir. 
* **Metin Tokenizasyonu:** `openai/clip-vit-large-patch14` tokenizer'ı kullanılır.

## 4. Dosya Yapısı ve Görevleri

### `dataset.py`
Ham verileri SD modelinin anlayacağı PyTorch `DataLoader` formatına dönüştürür.
* `extract_impression()`: Raporlardan "Impression" bölümünü regex ile çeker.
* `build_metadata()`: CSV ve görüntü dizinlerini tarayarak uygun PA görüntülerini ve optimize edilmiş rapor metinlerini eşleştirir. Uzun metinleri filtreler ve hız kazanmak adına sadece bir kere tokenization işlemi yaparak önbelleğe (cache) kaydeder.
* `MIMICCXRDataset`: PyTorch `Dataset` sınıfından türetilir. Diskten görüntüleri okur, `512x512`'ye boyutlandırır, normalize eder ([-1, 1] aralığına) ve CLIP token ID'leri ile birlikte döndürür.
* **Hafifletilmiş/Demo Opsiyonları:** Çok uzun sürebilecek eğitimleri önlemek için `max_samples` veya `use_augmented` gibi esnek parametreler içerir.

### `train.py`
Stable Diffusion modelinin ince ayar (fine-tuning) döngüsünün koşturulduğu ana kısımdır.
* **Model Kurulumu:** VAE, U-Net, CLIP ve Scheduler bileşenlerini HF Hub üzerinden çeker. Cihaz kapasitesine göre `bfloat16` veya `float32` hassasiyetini ayarlar.
* **Bellek Optimizasyonu (Gradient Checkpointing / Accumulation):** Çok büyük bir model olduğu için `gradient_checkpointing` kullanır. Ayrıca büyük GPU gereksinimini dengelemek için `gradient_accumulation_steps` (örn. 256 batch size hedefine ulaşmak için per_device_batch_size: 1 ve grad_accum: 256) parametrelerini destekler.
* **Demo Mode (`--demo_mode`):** Kodu veya bilgisayarı test etmek isteyenler için çok daha düşük çözünürlük (256x256), az sayıda örnek ve sınırlı adımdan oluşan güvenlik/test modu sunar.
* **Kontrol Noktası Kaydetme:** Her `save_every` adımında `unet/`, `text_encoder/` klasörlerini standart diffusers formatında `checkpoints/` altına kaydeder.

### `inference.py`
Eğitilmiş ağı kullanarak dışarıdan verilen yeni tıbbi prompt'lar için röntgen üretilmesini sağlar.
* **Scheduler:** Varsayılan olarak makalenin belirttiği `PNDMScheduler` ayarlanır.
* **Ayarlar:** `75` çıkarım adımı (inference steps) ve `4.0` CFG Scale (Classifier-Free Guidance Scale) ile yüksek kaliteli çıktılar (512x512) üretmeye uygun tasarlanmıştır.
* `generate_and_save()`: Üretilen PNG görselleri metindeki kelimelere göre normalize edilmiş dosya isimleriyle diskteki (`generated/`) dizinine yazar.

### `requirements.txt` ve Ortam
Projeyi makalenin birebir şartlarında tekrar üretebilmek için hassas sürümler seçilmiştir:
* `diffusers==0.7.1` ve Hugging Face bağımlılıkları (`transformers==4.25.1`).
* Orijinal ekosistemle kütüphane çakışması olmaması adına PyTorch `1.12.1` sürümü hedeflenir (CUDA donanımına göre doğru sürümün indirilmesi şarttır).

## 5. Pratik Kullanım Senaryosu / Projeyi Ayağa Kaldırma

**1. Çevresel Kurulum:** Python sanal ortamında `requirements.txt` kurun (GPU'nuza uygun cudatoolkit ayarlayarak).
**2. Hızlı Test (Küçük sistemler / İlk çalıştırma):** 
`python train.py --demo_mode` 
(Kodun hatasız çalışıp çalışmadığını anlamak için idealdir)
**3. Gerçek Eğitim:**
`python train.py --image_size 512 --max_train_steps 60000 --learning_rate 5e-5`
**4. Üretim / Çıkarım (Training sonrası test):**
`python inference.py --checkpoint_dir ./checkpoints/checkpoint-60000 --prompt "Right lower lobe pneumonia"`

## 6. Sonuç
Bu proje yapısı, akademik bir yayın standartlarını, üretim esnekliğiyle birleştiren oldukça temiz bir kod dizilimine (pipeline) sahiptir. Stable Diffusion tabanlı her türlü medikal *fine-tuning* senaryosunda baz alınabilecek kadar modüler dizayn edilmiştir.