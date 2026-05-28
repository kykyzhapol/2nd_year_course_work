"""
Модуль для визуализации LISA-кластеров (локальный индекс Морана) на карте России.
Module for visualizing LISA clusters (Local Moran's I) on the map of Russia.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re

# Импортируем файл локализации (словарь соответствий названий регионов)
# Import localization file (dictionary of region name mappings)
import mapping_config


# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ LISA
# 1. LOADING AND PREPARING LISA DATA
# ==========================================

# Загружаем файл результатов анализа по регионам
# Load the regional analysis results file
file_name = 'spatial_analysis_results.xlsx'
df_results = pd.read_excel(file_name)


def clean_lisa_quadrant(text):
    """
    Преобразует текстовое описание квадранта в стандартный класс LISA.
    Converts quadrant text description to standard LISA class.

    Args:
        text: Исходный текст / Original text

    Returns:
        str: Стандартное название квадранта / Standard quadrant name
    """
    # Проверка на пустые значения / Check for empty values
    if pd.isna(text):
        return 'Not Significant'

    text = str(text)

    # Проверяем наличие ключевых слов / Check for keywords
    if 'High-High' in text:
        return 'High-High'
    if 'Low-Low' in text:
        return 'Low-Low'
    if 'High-Low' in text:
        return 'High-Low'
    if 'Low-High' in text:
        return 'Low-High'

    # Если ничего не найдено / If nothing found
    return 'Not Significant'


# Обрабатываем оба года исследования / Process both study years
df_results['LISA_2004'] = df_results['Quadrant_2004'].apply(clean_lisa_quadrant)
df_results['LISA_2024'] = df_results['Quadrant_2024'].apply(clean_lisa_quadrant)

# Убираем возможные скрытые пробелы в названиях регионов
# Remove possible hidden spaces in region names
df_results['Регион'] = df_results['Регион'].str.strip()


def normalize_for_merge(name):
    """
    Нормализует название региона для бронебойного слияния.
    Normalizes region name for robust merging.

    Убирает пробелы, точки, дефисы, регистр и стандартные суффиксы субъектов РФ.
    Removes spaces, dots, hyphens, case, and standard suffixes of Russian subjects.

    Args:
        name: Исходное название региона / Original region name

    Returns:
        str: Нормализованное название для слияния / Normalized name for merging
    """
    # Проверка на не-строковые значения / Check for non-string values
    if not isinstance(name, str):
        return ''

    # Приводим к нижнему регистру и удаляем лишние пробелы
    # Convert to lowercase and strip extra spaces
    name = name.lower().strip()

    # Удаляем любые символы, кроме букв и цифр (убирает точки, дефисы, пробелы)
    # Remove any characters except letters and digits (removes dots, hyphens, spaces)
    name = re.sub(r'[^а-яёa-z0-9]', '', name)

    # Удаляем стандартные приставки/окончания субъектов РФ
    # Remove standard prefixes/suffixes of Russian subjects
    for word in ['республика', 'область', 'край', 'автономныйокруг', 'ао', 'город', 'г', 'кузбасс']:
        name = name.replace(word, '')

    # Точечные исправления популярных расхождений в словарях
    # Point fixes for common discrepancies in dictionaries
    if 'петербург' in name or 'спб' in name:
        return 'санктпетербург'
    if 'москва' in name:
        return 'москва'
    if 'осетия' in name or 'алания' in name:
        return 'осетия'
    if 'татарстан' in name:
        return 'татарстан'
    if 'башкортостан' in name or 'башкир' in name:
        return 'башкортостан'
    if ('саха' in name and 'якут' in name) or 'якути' in name:
        return 'якутия'
    if 'чуваш' in name:
        return 'чувашия'
    if 'ханты' in name or 'хмао' in name or 'югра' in name:
        return 'хмао'
    if 'ямало' in name or 'янао' in name:
        return 'янао'

    return name


# --- НОВАЯ УМНАЯ ФУНКЦИЯ ЗАПОЛНЕНИЯ АО ---
# --- NEW SMART FUNCTION FOR FILLING AUTONOMOUS OKRUGS ---
def fix_all_aos(gdf):
    """
    Безопасное копирование данных материнских регионов в автономные округа.
    Safely copies data from parent regions to autonomous okrugs.

    Использует строгие правила фильтрации, чтобы "Ненецкий" не путался с "Ямало-Ненецким".
    Uses strict filtering rules to prevent "Nenets" from being confused with "Yamalo-Nenets".

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame с данными LISA
                                      GeoDataFrame with LISA data

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame с исправленными данными АО
                                GeoDataFrame with fixed AO data
    """
    # Правила для автономных округов / Rules for autonomous okrugs
    # ao_keyword: что ищем / what to search for
    # exclude: что исключаем / what to exclude
    # parent: регион-донор данных / parent region (data donor)
    ao_rules = [
        {'ao_keyword': 'Ненецкий', 'exclude': 'Ямало', 'parent': 'Архангельская'},
        {'ao_keyword': 'Ханты-Мансийский', 'exclude': None, 'parent': 'Тюменская'},
        {'ao_keyword': 'Ямало-Ненецкий', 'exclude': None, 'parent': 'Тюменская'},
        {'ao_keyword': 'Чукотский', 'exclude': None, 'parent': 'Магаданская'}
    ]

    # Обрабатываем оба года / Process both years
    for year in ['2004', '2024']:
        col = f'LISA_{year}'

        for rule in ao_rules:
            # 1. Ищем материнский регион / Find parent region
            parent_mask = gdf['russian_name'].str.contains(rule['parent'], na=False)
            if not parent_mask.any():
                continue
            parent_idx = gdf[parent_mask].index[0]
            parent_val = gdf.loc[parent_idx, col]

            # 2. Ищем АО (с учетом исключений) / Find AO (considering exclusions)
            ao_mask = gdf['russian_name'].str.contains(rule['ao_keyword'], na=False)
            if rule['exclude']:
                ao_mask = ao_mask & ~gdf['russian_name'].str.contains(rule['exclude'], na=False)

            if not ao_mask.any():
                continue
            ao_idx = gdf[ao_mask].index[0]

            # 3. Переписываем значение, если оно пустое или Not Significant
            # Copy value if it's empty or Not Significant
            current_val = gdf.loc[ao_idx, col]
            if pd.isna(current_val) or current_val == 'Not Significant':
                gdf.loc[ao_idx, col] = parent_val

    return gdf
# ----------------------------------------


# Создаем нормализованный ключ для слияния в таблице результатов
# Create normalized merge key in results table
df_results['match_key'] = df_results['Регион'].apply(normalize_for_merge)

# Защита от пустых ключей / Protection against empty keys
df_results.loc[df_results['match_key'] == "", 'match_key'] = df_results['Регион']


# ==========================================
# 2. ЗАГРУЗКА КАРТЫ И МЭТЧИНГ
# 2. LOADING MAP AND MATCHING
# ==========================================

# Загружаем файл геоданных / Load geodata file
gdf_map = gpd.read_file('russia_regions.geojson')
gdf_map['name'] = gdf_map['name'].str.strip()

# Добавляем в геоданные русские названия регионов на основе подгруженного словаря
# Add Russian region names to geodata based on loaded dictionary
gdf_map['russian_name'] = gdf_map['name'].map(mapping_config.REGION_MAPPING)

# Создаем нормализованный ключ для слияния в геоданных
# Create normalized merge key in geodata
gdf_map['match_key'] = gdf_map['russian_name'].apply(normalize_for_merge)

# Защита: если регион не смапился, даем ему оригинальное имя
# Protection: if region not mapped, give it original name
gdf_map.loc[gdf_map['match_key'] == "", 'match_key'] = gdf_map['name']

# Объединяем карту и таблицу результатов по ключу 'match_key'
# Merge map and results table using 'match_key'
merged_gdf = gdf_map.merge(df_results, on='match_key', how='left')

# Применяем умный фикс для автономных округов
# Apply smart fix for autonomous okrugs
merged_gdf = fix_all_aos(merged_gdf)

# Заполняем пропуски (для регионов без данных или значимости)
# Fill missing values (for regions without data or significance)
merged_gdf['LISA_2004'] = merged_gdf['LISA_2004'].fillna('Not Significant')
merged_gdf['LISA_2024'] = merged_gdf['LISA_2024'].fillna('Not Significant')


# Выводим статистику по кластерам / Print cluster statistics
print("=== Статистика по кластерам LISA ===")
print("=== LISA Cluster Statistics ===")
for year in ['2004', '2024']:
    print(f"\n{year} год / {year}:")
    cluster_counts = merged_gdf[f'LISA_{year}'].value_counts()
    for cluster_type, count in cluster_counts.items():
        print(f"  {cluster_type}: {count} регионов / regions")
print("=========================================\n")


# ==========================================
# 3. НАСТРОЙКА КЛАССИЧЕСКОЙ ПАЛИТРЫ LISA
# 3. CONFIGURING CLASSIC LISA COLOR PALETTE
# ==========================================

# Стандартные цвета Anselin Local Moran's I (GeoDa)
# Standard Anselin Local Moran's I colors (GeoDa)
color_palette = {
    'High-High': '#e31a1c',      # Ярко-красный (Горячие точки / Ядра роста)
                                 # Bright red (Hot spots / Growth cores)
    'Low-Low': '#1f78b4',        # Синий (Холодные точки / Периферия)
                                 # Blue (Cold spots / Periphery)
    'High-Low': '#fdbf6f',       # Оранжевый (Аномалия: Высокий среди низких)
                                 # Orange (Anomaly: High among low)
    'Low-High': '#a6cee3',       # Голубой (Аномалия: Низкий среди высоких)
                                 # Light blue (Anomaly: Low among high)
    'Not Significant': '#e0e0e0' # Серый (Статистически незначимые различия)
                                 # Gray (Statistically insignificant)
}


# ==========================================
# 4. ФУНКЦИЯ ОТРИСОВКИ И СОХРАНЕНИЯ КАРТ
# 4. FUNCTION FOR PLOTTING AND SAVING MAPS
# ==========================================

def plot_and_save_lisa(year_column, year_title):
    """
    Отрисовывает и сохраняет карту LISA-кластеров для указанного года.
    Plots and saves LISA cluster map for the specified year.

    Args:
        year_column (str): Название колонки с данными LISA / LISA data column name
        year_title (str): Название года для заголовка и имени файла / Year title for caption and filename
    """
    # Создаем фигуру и оси / Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    # Заголовок для научной работы / Title for academic paper
    ax.set_title(
        f'Карта пространственной кластеризации LISA субъектов РФ ({year_title} г.)\n'
        f'LISA Spatial Clustering Map of Russian Regions ({year_title})',
        fontsize=16,
        pad=20
    )

    # Рисуем карту по слоям (чтобы цвета легли строго по нашей палитре)
    # Draw map by layers (to apply colors strictly according to palette)
    for cluster_type, color in color_palette.items():
        subset = merged_gdf[merged_gdf[year_column] == cluster_type]
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=color,
                edgecolor='#ffffff',  # Белые границы регионов смотрятся аккуратнее
                                     # White region borders look neater
                linewidth=0.6
            )

    # Убираем оси с градусной сеткой / Remove axes with grid lines
    ax.axis('off')

    # Создаем легенду / Create legend
    legend_patches = [
        Patch(facecolor=col, edgecolor='#ffffff', label=name)
        for name, col in color_palette.items()
    ]

    # Добавляем легенду на карту / Add legend to the map
    ax.legend(
        handles=legend_patches,
        title="Типы кластеров / Cluster Types",
        loc="lower left",
        fontsize=11,
        title_fontsize=12
    )

    # Сохраняем картинку в высоком качестве / Save high-quality image
    output_image_name = f'LISA_Map_Russia_{year_title}.png'
    plt.savefig(output_image_name, dpi=300, bbox_inches='tight')
    print(f"✓ Успешно сохранено: {output_image_name}")
    print(f"✓ Successfully saved: {output_image_name}")

    # Отображаем карту / Display the map
    plt.show()


# ==========================================
# 5. ЗАПУСК ГЕНЕРАЦИИ КАРТ
# 5. RUNNING MAP GENERATION
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ КАРТ LISA-КЛАСТЕРОВ / LISA CLUSTER MAPS GENERATION")
    print("="*60 + "\n")

    # Запускаем генерацию для обоих периодов / Run generation for both periods
    plot_and_save_lisa('LISA_2004', '2004')
    plot_and_save_lisa('LISA_2024', '2024')

    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА / GENERATION COMPLETED")
    print("="*60)