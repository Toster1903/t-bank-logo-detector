# t-bank-logo-detector

Проект для детекции логотипов (демо-пайплайн для отбора в программу Т-Банк). Работа не была полностью завершена, но показаны основные этапы подготовки данных, обучения моделей и REST API.

## Что делалось
- Отбор изображений по желтой компоненте с помощью OpenCV.
- Аугментации: шумы, повороты, негатив и др.
- Ручная разметка ~400 изображений через Label Studio.
- Дообучение YOLO для разметки остальных «желтых» изображений.
- Дообучение Faster R-CNN с backbone ResNet50 + FPN.
- Реализован FastAPI эндпоинт `/detect` для выгрузки координат предсказаний.

## Структура репозитория
- `yellow_detector.py` — фильтрация по желтому цвету.
- `noise.py` — аугментации изображений.
- `pars.py` — парсинг датасета.
- `model_train.ipynb` — обучение моделей.
- `yolov8s.pt`, `yolo11n.pt` — веса YOLO.
- `data.yaml` — конфигурация датасета.
- `endpoint.py` — FastAPI сервис.
- `requirements.txt` — зависимости.
- `runs/` — логи и артефакты тренировок.



Запуск через Docker :
git clone https://github.com/Toster1903/t-bank-logo-detector.git
cd t-bank-logo-detector

Собрать Docker образ:
docker build -t t-bank-logo-detector .

Запустить контейнер:
docker run -p 8000:8000 t-bank-logo-detector

Перейдите в браузере по ссылке:
http://127.0.0.1:8000/docs#/default/detect_logo_detect_post
