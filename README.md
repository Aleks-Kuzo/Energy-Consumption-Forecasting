# Energy-Consumption-Forecasting

# ⚡ Energy Consumption Forecasting (PyTorch LSTM)

Прогнозирование почасового энергопотребления производственного участка на основе временных рядов с помощью рекуррентной нейросети (LSTM), реализованной на PyTorch.

Проект объединяет два классических подхода к анализу временных рядов:
- **EDA и Feature Engineering** — в духе методики Rob Mulla (календарные признаки из индекса даты).
- **Sliding Window для глубокого обучения** — концепция M. Faaris, переписанная с TensorFlow/Keras на **PyTorch**.

## 📊 Датасет

[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (`PJME_hourly.csv`) — почасовые данные энергопотребления региона PJM East (США), опубликованные на Kaggle. Если файл датасета не найден локально, ноутбук автоматически генерирует синтетический временной ряд с суточной и годовой сезонностью — это позволяет проверить работоспособность пайплайна без скачивания данных.

## 🔧 Пайплайн

1. **Загрузка и сортировка данных** по временному индексу.
2. **Feature Engineering** — извлечение календарных признаков (`hour`, `dayofweek`, `quarter`, `month`, `year`, `dayofyear`).
3. **Train/test split по дате** (без случайного перемешивания, чтобы не допустить утечки данных).
4. **Масштабирование** через `MinMaxScaler`, обучаемый строго на train-выборке; отдельный scaler для целевой переменной для удобного `inverse_transform`.
5. **Sliding window** — кастомный `torch.utils.data.Dataset`, нарезающий ряд на окна фиксированной длины (по умолчанию 24 часа) для подачи в LSTM.
6. **Модель** — `LSTMForecaster`: многослойный LSTM + полносвязный выходной слой, предсказывающий следующее значение по последнему скрытому состоянию.
7. **Обучение** — MSELoss, оптимизатор Adam.
8. **Оценка** — MAE, RMSE, MAPE на тестовой выборке (после обратного масштабирования в МВт).
9. **Визуализация** — сопоставление факта и прогноза на графике.

## 🧠 Архитектура модели

```
Input (batch, window_size=24, n_features=7)
        │
        ▼
     LSTM (hidden_size=64, num_layers=2, dropout=0.2)
        │
        ▼
  Linear(64 → 1)  ← берётся выход последнего шага последовательности
        │
        ▼
   Прогноз (МВт)
```

## 🛠 Стек технологий

- **Python 3.13**
- **PyTorch** — построение и обучение LSTM
- **Pandas / NumPy** — обработка данных
- **Scikit-learn** — масштабирование и метрики
- **Matplotlib** — визуализация

## 🚀 Как запустить

```bash
git clone https://github.com/Aleks-Kuzo/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook energy_consumption_forecasting_pytorch.ipynb
```

Скачайте `PJME_hourly.csv` с [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) и укажите путь к файлу в соответствующей ячейке ноутбука (или запустите без файла — сработает демо-режим на синтетических данных).

### requirements.txt

```
numpy
pandas
matplotlib
scikit-learn
torch
jupyter
```

## 📁 Структура репозитория

```
.
├── energy_consumption_forecasting_pytorch.ipynb   # основной ноутбук
├── requirements.txt
└── README.md
```

## 💡 Возможные направления развития

- Увеличение числа эпох обучения и подбор гиперпараметров (размер окна, hidden_size, число слоёв).
- Добавление лаговых и скользящих признаков (rolling mean/std).
- Сравнение с бейзлайнами (SARIMA, Prophet, Gradient Boosting) на тех же метриках.
- Учёт выходных/праздничных дней как отдельного признака.
- Экспорт обученной модели и инференс-скрипт для продакшена.
