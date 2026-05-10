# Deep Learning Anomaly Detection

Bu proje, zaman serisi anomali tespiti için Siyah Kutu (Deep Learning) ve Açıklanabilir (Otomata) modelleri karşılaştırmayı amaçlamaktadır. Proje, SOLID ve Nesne Yönelimli Programlama (OOP) prensiplerine sadık kalınarak geliştirilmiştir.

## Geliştirme Ortamı Kurulumu

Bu proje grup çalışmasına uygun şekilde standartlaştırılmıştır. Çakışmaları ve format farklılıklarını önlemek için aşağıdaki adımları takip ederek geliştirme ortamınızı kurun.

### 1. Virtual Environment (Sanal Ortam) Oluşturma ve Aktifleştirme
Projeyi klonladıktan sonra bir Python sanal ortamı oluşturun:

**Windows için:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux için:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Bağımlılıkların Yüklenmesi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Kod Formatlama ve Linting (Ruff)
Projede kod formatlayıcı ve linter olarak **Ruff** kullanılmaktadır. Ayarlar `pyproject.toml` ve `.vscode/settings.json` içinde tanımlıdır. VSCode kullanıyorsanız dosyayı kaydettiğinizde otomatik olarak çalışacaktır. Manuel çalıştırmak için:
```bash
ruff check .
ruff format .
```

### 4. Testleri Çalıştırma
Projede test yazmak ve çalıştırmak için **pytest** kullanılmaktadır. GitHub Actions (CI) sürecinden geçebilmesi için testlerin başarılı olması gereklidir:
```bash
pytest
```

## Proje Klasör Yapısı
* `src/`: Ana kaynak kod (core, data, models, orchestration, reporting, visualization modülleri).
* `tests/`: Pytest ile yazılmış birim testleri.
* `config/`: Hard-coded değerleri engellemek için tüm parametrelerin bulunduğu konfigürasyon dosyaları.
* `venv/`: (Git tarafından yoksayılır) Python sanal ortamı.
* `data_raw/`: (Git tarafından yoksayılır) SKAB ve BATADAL veri setleri.

Herhangi bir sorun yaşarsanız `ci.yml` (GitHub Actions) çıktılarını kontrol edin ve testleri lokalinizde başarıyla geçirdiğinizden emin olun.
