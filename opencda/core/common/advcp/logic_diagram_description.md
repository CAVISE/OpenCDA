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
            modified_data, defense_score, metrics = advcp_manager.process_tick(i, batch_data, original_pred)

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

def process_tick(self, tick_number: int, batch_data: Optional[Dict] = None,
                  predictions: Optional[Dict] = None) -> Tuple[Optional[Dict], Optional[float], Optional[Dict]]:
    """Обрабатывает тик без circular dependency"""
    if not self.with_advcp:
        return None, None, None

    # Определяем этап атаки и подготавливаем данные соответствующим образом
    if self.attack_type in ["lidar_remove_late", "lidar_spoof_late"]:
        # Late атаки требуют и сырые данные, и предсказания
        raw_data = self._get_coperception_data(tick_number)
        if raw_data is None:
            return None, None, None

        if predictions is None:
            logger.error("Late атаки требуют predictions, но они не предоставлены")
            return None, None, None

        # Применяем late атаку к предсказаниям
        modified_predictions = self._apply_attack(raw_data, predictions, tick_number)

        # Применяем защиту если включена
        if self.apply_cad_defense and self.defender:
            defended_data, defense_score, defense_metrics = self._apply_defense(
                raw_data, modified_predictions, tick_number
            )
            return defended_data, defense_score, defense_metrics

        return modified_predictions, None, None
    else:
        # Early/intermediate атаки требуют СЫРЫЕ данные (не preprocessed batch_data)
        raw_data = self._get_coperception_data(tick_number)
        if raw_data is None:
            return None, None, None

        # Применяем атаку к сырым данным
        attacked_data = self._apply_attack(raw_data, tick_number)

        # Конвертируем атакованные сырые данные обратно в формат OpenCOOD
        if self.perception is not None:
            ego_id = self._get_ego_vehicle_id(raw_data)

            # Применяем соответствующий препроцессор в зависимости от типа атаки
            if self.attack_type in ["lidar_remove_early", "lidar_spoof_early"]:
                preprocessed_data = self.perception.early_preprocess(attacked_data, ego_id)
            elif self.attack_type in ["lidar_remove_intermediate", "lidar_spoof_intermediate"]:
                preprocessed_data = self.perception.intermediate_preprocess(attacked_data, ego_id)
            else:
                preprocessed_data = None

            if preprocessed_data is not None:
                # Применяем защиту если включена
                if self.apply_cad_defense and self.defender:
                    preprocessed_data, defense_score, defense_metrics = self._apply_defense(
                        preprocessed_data, tick_number
                    )
                    return preprocessed_data, defense_score, defense_metrics

                return preprocessed_data, None, None

        return None, None, None
```

## Структура AdvCPManager

### 0. OpencoodPerception (Preprocessing Component)

**Назначение:** Препроцессинг сырых данных в формат OpenCOOD

**Методы:**
- `early_preprocess()` - для early fusion атак
- `intermediate_preprocess()` - для intermediate fusion атак
- `late_preprocess()` - для late fusion атак

**Использование:**
- Инициализируется при включении AdvCP
- Используется для конвертации атакованных сырых данных обратно в формат OpenCOOD
- Требуется для early/intermediate атак после применения атаки

### 1. Атаки (Attack Engine)

**Выбор злоумышленников:**

- На основе `attackers_ratio` (например, 0.2 = 20% машин)
- Стратегии: `random`, `all_non_attackers`

**Типы атак по уровню:**

| Уровень | Атака | Что модифицирует | Требуемые данные | Инструменты |
| --- | --- | --- | --- | --- |
| **Early** | Удаление/Спуфинг точек | Сырые LiDAR точки | Сырые данные | Ray Casting Engine |
| **Intermediate** | Удаление/Спуфинг фич | Промежуточные признаки | Сырые данные + Perception | Adversarial Generator |
| **Late** | Удаление/Спуфинг боксов | Финальные bounding boxes | Сырые данные + Предсказания | Прямая модификация |

**Пример работы атаки:**

```python
# Late атаки (требуют predictions)
Attacker.run(multi_frame_case, attack_opts):
    # multi_frame_case содержит сырые данные + predictions для ego vehicle
    for attacker_vehicle:
        # Модифицировать predictions (боксы/скоры)
        modified_predictions = {
            "pred_bboxes": ...,  # измененные боксы
            "pred_scores": ...,  # измененные скоры
        }
    return modified_predictions

# Early/Intermediate атаки (требуют только сырые данные)
Attacker.run(multi_frame_case, attack_opts):
    for attacker_vehicle:
        # Модифицировать сырые данные (точки/фичи)
        modified_data[vehicle_id] = {
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
def _apply_defense(self, data: Dict, tick_number: int) -> Tuple[Dict, Optional[float], Optional[Dict]]:
    """Применяет защиту к данным"""
    if not self.defender:
        return data, None, None

    # Подготовка multi-frame case для защиты
    multi_frame_case = {tick_number: data}
    defend_opts = {"frame_ids": [tick_number], "vehicle_ids": list(data.keys())}

    # Запуск защиты
    defended_data, score, metrics = self.defender.run(multi_frame_case, defend_opts)

    return defended_data[tick_number], score, metrics
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
            self.advcp_manager = AdvCPManager(opt, current_time, self, message_handler)

    def make_prediction(self, tick_number):
        for batch_data in self.data_loader:
            if self.advcp_manager and self.advcp_manager.with_advcp:
                # Оригинальный инференс
                original_pred = inference_utils.inference_xxx_fusion(batch_data, ...)

                # AdvCP обработка (передаем batch_data и predictions)
                modified_data, defense_score, metrics = self.advcp_manager.process_tick(
                    tick_number, batch_data, original_pred
                )

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
