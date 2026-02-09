## Обзор

Модуль **AdvCP (Advanced Collaborative Perception)** добавляет возможность проведения атак и применения защит на систему совместного восприятия автономных транспортных средств. Интегрирован в `CoperceptionModelManager` как опциональный компонент.

## Основной поток данных

```
CARLA → DataLoader → CoperceptionModelManager → AdvCPManager → Модифицированные предсказания → Оценка
```

### Ключевой метод

```python
CoperceptionModelManager.make_prediction(tick_number):
    for i, batch_data in enumerate(self.data_loader):
        # Сохраняем текущий batch_data для AdvCP (избегаем circular dependency)
        self._current_batch_index = i
        self._current_batch_data = batch_data

        if with_advcp:
            # 1. Получить оригинальные предсказания
            original_pred = inference_utils.inference_xxx_fusion()

            # 2. Применить AdvCP (без рекурсии)
            modified_data, defense_score, metrics = advcp_manager.process_tick(i)

            # 3. Использовать модифицированные данные
            pred_box_tensor = extract_from(modified_data)
        else:
            # Обычный инференс
            pred_box_tensor, pred_score, gt = inference_utils.inference_xxx_fusion()
```

### Методы для получения данных для advpc

```python
# В CoperceptionModelManager
def _get_raw_data(self, tick_number: int) -> Optional[Dict]:
    """Возвращает сырые данные без запуска предсказания"""
    if self._current_batch_index == tick_number and self._current_batch_data is not None:
        return self._current_batch_data
    # Fallback: доступ к датасету напрямую
    if tick_number < len(self.opencood_dataset):
        return self.opencood_dataset[tick_number]
    return None

# В AdvCPManager
def _get_coperception_data(self, tick_number: int) -> Dict:
    """Получает данные напрямую без вызова make_prediction()"""
    raw_data = self.coperception_manager._get_raw_data(tick_number)
    if raw_data is None:
        logger.warning(f"No raw data for tick {tick_number}")
        return {}
    return raw_data

def process_tick(self, tick_number: int) -> Tuple[Optional[Dict], Optional[float], Optional[Dict]]:
    """Обрабатывает тик без circular dependency"""
    if not self.with_advcp:
        return None, None, None
    # Получаем данные напрямую, а не через make_prediction()
    coperception_data = self._get_coperception_data(tick_number)
    # ... остальной код атаки/защиты
```

## Структура AdvCPManager

### 1. Атаки (Attack Engine)

**Выбор злоумышленников:**

- На основе `attackers_ratio` (например, 0.2 = 20% машин)
- Стратегии: `random`, `all_non_attackers`

**Типы атак по уровню:**

| Уровень | Атака | Что модифицирует | Инструменты |
| --- | --- | --- | --- |
| **Early** | Удаление/Спуфинг точек | Сырые LiDAR точки | Ray Casting Engine |
| **Intermediate** | Удаление/Спуфинг фич | Промежуточные признаки | Adversarial Generator |
| **Late** | Удаление/Спуфинг боксов | Финальные bounding boxes | Прямая модификация |

**Пример работы атаки:**

```python
Attacker.run(multi_vehicle_case, attack_opts):
    for attacker_vehicle:
        # Модифицировать данные (точки/фичи/боксы)
        modified_data[vehicle_id] = {
            "pred_bboxes": ...,  # измененные боксы
            "pred_scores": ...,  # измененные скоры
            "lidar": ...,       # измененные точки (early)
            "features": ...     # измененные фичи (intermediate)
        }
    return modified_data
```

### 2. Защита CAD (Defense Engine)

**Принцип:** Проверка геометрической консистентности между предсказаниями и occupancy map.

**Шаги:**

1. **Построение occupancy map:**
    - Сегментация LiDAR точек → объекты
    - Обнаружение земли → удаление
    - Трекинг → временная консистентность
    - Объединение данных от всех машин
2. **Проверка консистентности:**
    - **Спуфинг:** предсказанный бокс в свободной области → подозрение
    - **Удаление:** занятая область без предсказания → пропущенный объект
3. **Возврат:**

```python
defended_data, score, metrics = defender.run(multi_frame_case, opts)
```

**Инструменты защиты:**

- Occupancy Map Generator
- Ground Detection
- Segmentation
- Object Tracking
- Consistency Checker

## Интеграция с OpenCOOD

### CoperceptionModelManager

```python
class CoperceptionModelManager:
    def __init__(self, opt, ...):
        # Загрузка модели OpenCOOD
        self.model = train_utils.create_model(hypes)
        _, self.model = train_utils.load_saved_model(...)

        # Создание DataLoader
        self.opencood_dataset = build_dataset(hypes, ...)
        self.data_loader = DataLoader(...)

        # Инициализация AdvCP (если включен)
        if opt.get("with_advcp"):
            self.advcp_manager = AdvCPManager(advcp_config, ...)

    def make_prediction(self, tick_number):
        for batch_data in self.data_loader:
            if self.advcp_manager and self.advcp_manager.with_advcp:
                # Оригинальный инференс
                original_pred = inference_utils.inference_xxx_fusion(batch_data, ...)

                # AdvCP обработка
                modified_data, defense_score, metrics = self.advcp_manager.process_tick(tick_number)

                # Извлечение предсказаний
                if modified_data:
                    pred_box_tensor = extract_predictions(modified_data)
                else:
                    pred_box_tensor = original_pred[0]
            else:
                # Обычный инференс
                pred_box_tensor, pred_score, gt = inference_utils.inference_xxx_fusion(...)

            # Оценка и визуализация
            eval_utils.caluclate_tp_fp(...)
```

## Конфигурация

### CLI параметры

```bash
--with-advcp                    # Включить AdvCP
--attack-type <type>            # Тип атаки
--attackers-ratio <0.0-1.0>     # Доля атакующих машин
--attack-target <strategy>      # Стратегия выбора целей
--apply-cad-defense             # Включить защиту CAD
--defense-threshold <float>     # Порог срабатывания
```

### Пример конфигурации

`advcp_config.yaml`:

```yaml
attack:
  type: "lidar_remove_early"
  ratio: 0.2
  target: "random"
  dense: 1

defense:
  enabled: true
  threshold: 0.7
```

## Типы атак подробно

### Early Fusion Attacks

- **LidarRemoveEarlyAttacker**: Удаляет точки целевого объекта с помощью ray tracing
- **LidarSpoofEarlyAttacker**: Добавляет точки fake-объекта (3D модель машины)

### Intermediate Fusion Attacks

- **LidarRemoveIntermediateAttacker**: Оптимизирует состязательное возмущение в feature space
- **LidarSpoofIntermediateAttacker**: Генерирует состязательные признаки

### Late Fusion Attacks

- **LidarRemoveLateAttacker**: Удаляет bounding box из предсказаний
- **LidarSpoofLateAttacker**: Добавляет fake bounding box

### AdvShapeAttacker

Универсальная атака, оптимизирующая форму 3D-модели через генетический алгоритм.

## Защита CAD подробно

### Алгоритм

```python
def run(self, multi_frame_case, opts):
    # 1. Сбор occupancy maps от всех машин
    occupied_areas = []
    free_areas = []
    for vehicle_id, vehicle_data in frame_data.items():
        occupied_areas += vehicle_data["occupied_areas"]
        free_areas.append(vehicle_data["free_areas"])
    free_areas = unary_union(free_areas)

    # 2. Сбор ground truth боксов
    gt_bboxes = collect_unique_gt_boxes(frame_data)

    # 3. Проверка консистентности
    metrics = run_core(pred_bboxes, gt_bboxes, occupied_areas, free_areas, ego_area)

    # 4. Возврат defended_data + scores
    return defended_data, score, metrics
```

### Критерии обнаружения

| Тип атаки | Признак | Действие |
| --- | --- | --- |
| Спуфинг | Бокс в свободной области | Низкий trust score |
| Удаление | Занятая область без бокса | Пометить как пропущенный объект |

## Файлы модуля

```
opencda/core/common/advcp/
├── advcp_manager.py          # Главный менеджер
├── advcp_config.py           # Загрузка конфига
├── advcp_config.yaml         # Конфиг
└── mvp/                      # Реализации
    ├── attack/               # 7 типов атак
    ├── defense/              # CAD защита
    ├── perception/           # Обертка OpenCOOD
    ├── data/                 # Датсеты и утилиты
    ├── tools/                # Ray tracing, seg, tracking
    └── visualize/            # Визуализация
```