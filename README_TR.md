# YOLO Vision API

[English](README.md) | [Türkçe](README_TR.md)

Ultralytics YOLO modelini tip güvenli REST uçları üzerinden sunan, görsel ve video kabul eden production-style FastAPI servisidir. Sınıf, güven skoru, koordinatlar ve hız metrikleri döndürür; tarayıcı arayüzünde sonuç kutularını gösterir.

## Özellikler

- Görsel yükleme ve tarayıcı üzerinde kutu çizimi
- Ayarlanabilir kare örneklemeli video analizi
- Sınıf kimliği, etiket, güven skoru ve bounding box sonuçları
- Ön işleme, inference, son işleme ve toplam süre metrikleri
- Video FPS, süre, işlenen kare ve toplam tespit metrikleri
- Resmî veya özel `.pt` ağırlıkları kullanabilme
- Dosya boyutu sınırı, medya doğrulama ve geçici dosya temizliği
- Docker, testler, Ruff ve GitHub Actions CI

> Güven skoru model doğruluğu değildir. mAP50-95 gibi gerçek doğruluk metrikleri, etiketli doğrulama veri setiyle YOLO validation veya benchmark çalıştırılarak ölçülmelidir.

## Docker ile çalıştırma

```bash
copy .env.example .env
docker compose up --build
```

Arayüz: `http://localhost:8001`  
Swagger: `http://localhost:8001/docs`

İlk inference sırasında resmî ağırlık dosyası indirileceği için başlangıç daha uzun sürebilir. Özel model kullanmak için `.env` içindeki `MODEL_PATH` değerini değiştirin.

## Yerel geliştirme ve test

```bash
python -m venv .venv
pip install -e ".[inference,dev]"
uvicorn app.main:app --reload --port 8001
ruff check .
pytest -q
```

## Sonraki adımlar

- Uzun videolar için arka plan iş kuyruğu
- İşlenmiş çıktıların S3 uyumlu depolamaya yazılması
- API anahtarı, rate limit ve Prometheus metrikleri
- Model sürüm kaydı ve etiketli veri seti doğruluk raporları
- ONNX/TensorRT performans karşılaştırması
