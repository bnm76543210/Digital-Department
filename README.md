# Проект: Продвинутая мультиклассовая классификация для предиктивного обслуживания оборудования

**Выпускная квалификационная работа**  
**Выполнил:** Фахрутдинов Марат Альбертович  
**Программа:** Цифровая кафедра - ИИ в машиностроении

## Описание проекта

Цель проекта - разработать модель машинного обучения для предиктивного обслуживания промышленного оборудования. Базовая постановка задачи определяет, произойдет ли отказ оборудования (`Machine failure = 1`) или нет (`Machine failure = 0`). В продвинутой версии проект расширен: реализована мультиклассовая классификация типа отказа, детальный анализ данных, улучшенная предобработка, сравнение и оптимизация моделей, а также интеграция с инструментами MLOps (`ClearML`).

Используется датасет `AI4I 2020 Predictive Maintenance Dataset`. Приложение помогает оценивать состояние оборудования по технологическим параметрам и прогнозировать не только сам факт отказа, но и его тип.

Классы целевой переменной:

- `No failure` - отказ не ожидается;
- `TWF` - отказ из-за износа инструмента;
- `HDF` - отказ из-за недостаточного отвода тепла;
- `PWF` - отказ из-за некорректной мощности процесса;
- `OSF` - отказ из-за перегрузки оборудования;
- `RNF` - случайный отказ;
- `Multiple failures` - одновременно отмечено несколько типов отказов.

## Выполненные задачи

### **1. Изменение задачи на мультиклассовую классификацию**

- **Описание:** реализована мультиклассовая классификация для предсказания типа отказа оборудования (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`, `Multiple failures`, `No failure`).
- **Инструменты:** `LabelEncoder`, `RandomForestClassifier`, `ExtraTreesClassifier`, `XGBoost` с `objective='multi:softprob'`.
- **Результаты:** при проверке на полном датасете лучшей моделью стала `XGBoost`:
  - Accuracy: **0.9855**
  - Weighted F1-score: **0.9820**

### **2. Детальный анализ датасета**

- **Описание:** добавлена отдельная страница Streamlit `Обзор данных` с подробным анализом исходного датасета.
- **Инструменты:** `pandas`, `seaborn`, `matplotlib`, `Streamlit`.
- **Реализовано:** просмотр фрагмента данных, распределение типов отказов и типов продукта, описательная статистика, гистограммы числовых признаков, корреляционная матрица и scatter plot для анализа взаимосвязей между признаками.

### **3. Улучшение предобработки и оптимизация обучения модели**

- **Описание:** предобработка вынесена в единый `Pipeline`, чтобы одинаково обрабатывать данные при обучении, локальном прогнозе и MLOps-сценариях.
- **Инструменты:** `ColumnTransformer`, `SimpleImputer`, IQR-обработка выбросов, `StandardScaler`, `OneHotEncoder`, `StratifiedKFold`, `GridSearchCV`.
- **Результаты:** реализованы K-Fold кросс-валидация, сравнение нескольких моделей и опциональный подбор гиперпараметров `RandomForestClassifier`.

### **4. Внедрение ClearML**

- **Описание:** интегрирован ClearML для отслеживания экспериментов, управления версиями датасета, сохранения модели и развертывания через ClearML Serving.
- **Инструменты:** `clearml.Task`, `clearml.Dataset`, `clearml.OutputModel`, `clearml-serving`, Docker.
- **Результаты:**
  - метрики и параметры обучения отправляются в ClearML Task;
  - датасет публикуется как ClearML Dataset;
  - лучшая модель загружается в ClearML Model Registry;
  - модель зарегистрирована в ClearML Serving;
  - предсказания выполняются через REST API.

## Структура проекта

```text
Digital-Department/
├── app.py
├── analysis_and_model.py
├── data_overview.py
├── clearml_integration.py
├── mlops_page.py
├── maintenance_core.py
├── serving_preprocess.py
├── presentation.py
├── requirements.txt
├── scripts/
│   ├── clearml_dataset.py
│   ├── clearml_train.py
│   └── serving_request.py
├── data/
│   └── ai4i2020.csv
└── tests/
    └── test_maintenance_core.py
```

## Установка

```powershell
git clone https://github.com/bnm76543210/Digital-Department.git
cd Digital-Department
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Развертывание с нуля

Для демонстрации полного сценария на чистой машине используйте такой порядок.

1. Склонировать репозиторий и перейти в папку проекта:

```powershell
git clone https://github.com/bnm76543210/Digital-Department.git
cd Digital-Department
```

2. Создать виртуальное окружение и установить зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. Настроить ClearML без сохранения ключей в репозитории:

```powershell
clearml-init
```

В открывшемся запросе вставьте блок `api { ... }` из личного кабинета ClearML. Ключи лучше не показывать крупным планом при записи экрана.

Альтернативный вариант через переменные окружения текущего терминала:

```powershell
$env:CLEARML_API_ACCESS_KEY="ВАШ_ACCESS_KEY"
$env:CLEARML_API_SECRET_KEY="ВАШ_SECRET_KEY"
$env:CLEARML_API_HOST="https://api.clear.ml"
$env:CLEARML_WEB_HOST="https://app.clear.ml"
$env:CLEARML_FILES_HOST="https://files.clear.ml"
```

4. Запустить приложение:

```powershell
streamlit run app.py
```

5. В Streamlit открыть `Модель и прогноз`, нажать `Обучить и сравнить`, затем открыть `MLOps ClearML` и нажать `Отправить метрики и модель в ClearML`.

6. При необходимости создать версию датасета и проверить CLI-сценарии:

```powershell
python scripts\clearml_dataset.py
python scripts\clearml_train.py
```

7. Для демонстрации Serving запустить Docker Desktop, зарегистрировать модель в ClearML Serving, поднять inference-контейнер и проверить REST-запрос:

```powershell
python scripts\serving_request.py --endpoint http://127.0.0.1:8080/serve/predictive_maintenance
```

## Запуск приложения

```powershell
streamlit run app.py
```

После запуска откроется Streamlit-интерфейс с четырьмя разделами:

- `Обзор данных` - EDA, распределения классов, корреляции и scatter plot;
- `Модель и прогноз` - обучение, сравнение моделей, матрицы ошибок и прогноз;
- `MLOps ClearML` - отправка эксперимента и модели в ClearML, создание Dataset, загрузка модели из ClearML и запрос к Serving endpoint;
- `Итоги проекта` - краткая презентация выполненных задач и метрик.

## ClearML

Секретные ключи не хранятся в репозитории. Настройте ClearML локально одним из двух способов.

Через интерактивную настройку:

```powershell
clearml-init
```

Или через переменные окружения PowerShell:

```powershell
$env:CLEARML_API_ACCESS_KEY="ВАШ_ACCESS_KEY"
$env:CLEARML_API_SECRET_KEY="ВАШ_SECRET_KEY"
$env:CLEARML_API_HOST="https://api.clear.ml"
$env:CLEARML_WEB_HOST="https://app.clear.ml"
$env:CLEARML_FILES_HOST="https://files.clear.ml"
```

Создать версию датасета:

```powershell
python scripts\clearml_dataset.py
```

Обучить модели и отправить Task, метрики и лучшую модель в ClearML:

```powershell
python scripts\clearml_train.py
```

Проверить ClearML Serving endpoint:

```powershell
python scripts\serving_request.py --endpoint http://127.0.0.1:8080/serve/predictive_maintenance
```

Пример регистрации модели в ClearML Serving:

```powershell
clearml-serving create --name "Predictive Maintenance Serving" --project "Predictive Maintenance Advanced"
clearml-serving --id <service_id> model add --engine sklearn --endpoint predictive_maintenance --model-id <model_id> --preprocess serving_preprocess.py --name "Predictive Maintenance Model" --project "Predictive Maintenance Advanced"
```

Запуск локального inference-контейнера:

```powershell
docker run -p 8080:8080 `
  -e CLEARML_API_ACCESS_KEY `
  -e CLEARML_API_SECRET_KEY `
  -e CLEARML_API_HOST=https://api.clear.ml `
  -e CLEARML_WEB_HOST=https://app.clear.ml `
  -e CLEARML_FILES_HOST=https://files.clear.ml `
  -e CLEARML_SERVING_TASK_ID=<service_id> `
  -e CLEARML_SERVING_POLL_FREQ=5 `
  -e CLEARML_EXTRA_PYTHON_PACKAGES="scikit-learn==1.6.1 xgboost==3.2.0 pandas==2.2.3" `
  allegroai/clearml-serving-inference:1.3.2
```

Для текущего проверенного запуска использовались:

- `service_id`: `a6a8ad8eac1f449fa66e9cc9067fbfe2`
- `model_id`: `1c9047b76b57474d960ff72fa4cc44c3`
- endpoint: `http://127.0.0.1:8080/serve/predictive_maintenance`

## Тестирование

```powershell
python -m unittest discover -s tests
```

Тесты проверяют формирование мультиклассовой цели, подготовку признаков, обучение базовых моделей, получение прогноза и безопасную диагностику ClearML.

## Датасет

Используется файл `data/ai4i2020.csv`. В приложении можно работать с ним сразу или загрузить CSV с такой же структурой через интерфейс.

## Видео-демонстрация

Папка `video/` добавлена в `.gitignore`: старый демонстрационный ролик не публикуется в репозиторий. Новое видео записывается после проверки финальной версии проекта и при необходимости добавляется отдельно.
