# Симуляция CFD реактора замедленного коксования

Python реализация CFD симуляции пилотной установки реактора замедленного коксования на основе статьи:

> Díaz, F. A., Chaves-Guerrero, A., Gauthier-Maradei, P., Fuentes, D., Guzmán, A., Picón, H. (2017). 
> CFD simulation of a pilot plant Delayed Coking reactor using an In-House CFD code. 
> CT&F - Ciencia, Tecnología y Futuro, 7(1), 85-100.

## Описание

Проект реализует трёхфазную (вакуумный остаток, дистилляты, кокс) динамическую CFD модель для симуляции процесса замедленного коксования. Модель включает:

- 2D осесимметричную геометрию
- Дискретизацию методом конечных объёмов
- Алгоритм SIMPLEC для связи давление-скорость
- Алгоритм PEA для учёта межфазного взаимодействия
- Многоступенчатую кинетику реакций
- Тепло- и массообмен между фазами

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/delayed-coking-cfd.git
cd delayed-coking-cfd

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка пакета
pip install -e .