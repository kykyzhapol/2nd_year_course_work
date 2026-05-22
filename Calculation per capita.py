"""
Модуль для расчета географических центров тяжести душевых экономических показателей регионов РФ.
Module for calculating geographical gravity centers of per capita economic indicators of Russian regions.

Загружает данные душевого ВРП и потребительской корзины, вычисляет веса регионов,
и определяет центры тяжести на сфере для 2004, 2014 и 2024 годов.
Loads per capita GRP and consumer basket data, calculates regional weights,
and determines spherical gravity centers for 2004, 2014, and 2024.
"""

import os
import re
import numpy as np
import pandas as pd
import folium
from math import radians, cos, sin, atan2, sqrt, degrees

# ==========================================
# ВСПОМОГАТЕЛЬНЫЙ СПРАВОЧНИК КООРДИНАТ РЕГИОНОВ РФ
# AUXILIARY DIRECTORY OF RUSSIAN REGION COORDINATES
# ==========================================
REGIONAL_COORDINATES = {
    # Центральный федеральный округ / Central Federal District
    'москва': (55.7558, 37.6173),
    'московская': (55.7115, 37.8920),
    'брянская': (52.8876, 33.4058),
    'владимирская': (56.0416, 40.4431),
    'ивановская': (56.9972, 41.6143),
    'тверская': (56.9942, 34.9048),
    'ярославская': (57.7314, 39.0534),

    # Приволжский федеральный округ / Volga Federal District
    'нижегородская': (56.2464, 44.5700),
    'татарстан': (55.4372, 50.5369),
    'башкортостан': (54.4078, 56.0271),

    # Уральский федеральный округ / Ural Federal District
    'свердловская': (58.4503, 61.2720),
    'челябинская': (54.7247, 61.0242),

    # Сибирский федеральный округ / Siberian Federal District
    'новосибирская': (55.2750, 79.9114),
    'красноярский': (64.2132, 98.4870),
    'иркутская': (56.4914, 103.5222),

    # Дальневосточный федеральный округ / Far Eastern Federal District
    'приморский': (45.1664, 134.1378),
    'хабаровский': (54.5126, 136.0028),
    'чукотский': (66.2573, 172.4332),
    'сахалинская': (50.6083, 142.7483),
    'магаданская': (62.8837, 151.7820),
    'амурская': (53.6667, 127.8333),
    'якутия': (66.3817, 124.9788),
    'бурятия': (53.1549, 109.8315),
    'забайкальский': (52.2033, 116.5167),
    'камчатский': (56.5100, 159.2600),

    # Северо-Западный федеральный округ / Northwestern Federal District
    'санкт-петербург': (59.9343, 30.3351),
    'ленинградская': (59.9645, 31.0267),
    'калининградская': (54.7671, 21.4682),
}


# ==========================================
# ФУНКЦИИ КЛИНИНГА И МАТЧИНГА НАЗВАНИЙ
# FUNCTIONS FOR CLEANING AND MATCHING REGION NAMES
# ==========================================
def normalize_region_name(name):
    """
    Нормализует название региона, удаляя лишние слова и приводя к стандартному виду.
    Normalizes region name by removing extra words and converting to standard form.

    Args:
        name (str): Исходное название региона / Original region name

    Returns:
        str: Нормализованное название / Normalized region name
    """
    # Проверка на не-строковые значения / Check for non-string values
    if not isinstance(name, str):
        return ""

    # Приводим к нижнему регистру и удаляем лишние пробелы
    # Convert to lowercase and strip extra spaces
    name = name.lower().strip()

    # Список слов для удаления из названий / List of words to remove from names
    replacements = [
        "республика", "область", "край", "г.", "город",
        "автономный округ", "ао", "авт.округ", "округ"
    ]

    # Удаляем лишние слова / Remove extra words
    for rep in replacements:
        name = name.replace(rep, "")

    # Нормализуем пробелы / Normalize spaces
    name = re.sub(r'\s+', ' ', name).strip()

    # Специальные случаи / Special cases
    if "саха" in name or "якутия" in name:
        return "якутия"
    if "северная осетия" in name:
        return "северная осетия"
    if "чувашия" in name or "чувашская" in name:
        return "чувашия"

    return name


def apply_valiulin_filter(df, region_col='Регион'):
    """
    Применяет фильтр Валиулина для исключения определённых регионов из анализа.
    Applies Valiulin filter to exclude certain regions from the analysis.

    Фильтр исключает:
    - Малые автономные округа (кроме Чукотки)
    - Регионы, исключенные автором из анализа
    - Технические строки

    Filter excludes:
    - Small autonomous districts (except Chukotka)
    - Regions excluded by the author from analysis
    - Technical rows

    Args:
        df (pd.DataFrame): DataFrame с данными / DataFrame with data
        region_col (str): Название колонки с регионами / Region column name

    Returns:
        pd.DataFrame: Отфильтрованный DataFrame / Filtered DataFrame
    """

    def is_valid(name):
        # Если в ячейке NaN или не строка — бракуем строку
        # If cell contains NaN or is not a string - reject the row
        if pd.isna(name) or not isinstance(name, str):
            return False

        n_lower = name.lower().strip()

        # Исключаем малые автономные округа (кроме Чукотки)
        # Exclude small autonomous districts (except Chukotka)
        if 'автономный округ' in n_lower or ' ао' in n_lower or n_lower.endswith(' ао'):
            if 'чукотский' not in n_lower:
                return False

        # Регионы, исключенные автором из анализа / Regions excluded by the author from analysis
        excluded_regions = [
            'крым', 'севастополь', 'чеченск', 'чечня',
            'донецк', 'луганск', 'запорож', 'херсон', 'днр', 'лнр'
        ]
        if any(excl in n_lower for excl in excluded_regions):
            return False

        # Исключаем технические строки / Exclude technical rows
        tech_rows = ['федеральный округ', 'российская федерация', 'всего', 'из суммы']
        if any(tech in n_lower for tech in tech_rows):
            return False

        return True

    # Применяем фильтр и возвращаем копию / Apply filter and return copy
    return df[df[region_col].apply(is_valid)].copy()


# ==========================================
# МАТЕМАТИЧЕСКИЙ РАСЧЕТ ЦЕНТРА ТЯЖЕСТИ НА СФЕРЕ
# MATHEMATICAL CALCULATION OF SPHERICAL GRAVITY CENTER
# ==========================================
def calculate_spherical_gravity_center(df, lat_col, lon_col, weight_col):
    """
    Вычисляет центр тяжести на сфере с учетом весовых коэффициентов.
    Calculates spherical gravity center considering weight coefficients.

    Использует перевод сферических координат (широта/долгота) в декартовы (X, Y, Z),
    усредняет с учетом весов, и конвертирует обратно в широту/долготу.
    Converts spherical coordinates (latitude/longitude) to Cartesian (X, Y, Z),
    averages with weights, and converts back to latitude/longitude.

    Args:
        df (pd.DataFrame): DataFrame с координатами и весами / DataFrame with coordinates and weights
        lat_col (str): Название колонки с широтой / Latitude column name
        lon_col (str): Название колонки с долготой / Longitude column name
        weight_col (str): Название колонки с весами / Weight column name

    Returns:
        tuple: (center_latitude, center_longitude) или (None, None) если сумма весов = 0
               (center_latitude, center_longitude) or (None, None) if total weight = 0
    """
    # Инициализация сумм для декартовых координат / Initialize sums for Cartesian coordinates
    X, Y, Z, total_w = 0.0, 0.0, 0.0, 0.0

    # Перебираем строки DataFrame / Iterate through DataFrame rows
    for _, row in df.iterrows():
        # Конвертируем градусы в радианы / Convert degrees to radians
        lat = radians(row[lat_col])
        lon = radians(row[lon_col])
        w = row[weight_col]

        # Пропускаем строки с невалидными весами / Skip rows with invalid weights
        if pd.isna(w) or w <= 0:
            continue

        # Перевод сферических координат в декартовы трехмерные с учетом веса
        # Convert spherical coordinates to weighted Cartesian 3D coordinates
        X += w * cos(lat) * cos(lon)
        Y += w * cos(lat) * sin(lon)
        Z += w * sin(lat)
        total_w += w

    # Проверка на нулевую сумму весов / Check for zero total weight
    if total_w == 0:
        return None, None

    # Нормализация декартовых координат / Normalize Cartesian coordinates
    X /= total_w
    Y /= total_w
    Z /= total_w

    # Обратный перевод декартовых координат в широту/долготу
    # Convert back from Cartesian to latitude/longitude
    center_lon = atan2(Y, X)
    center_lat = atan2(Z, sqrt(X ** 2 + Y ** 2))

    # Конвертируем радианы обратно в градусы / Convert radians back to degrees
    return degrees(center_lat), degrees(center_lon)


# ==========================================
# ОСНОВНОЙ ПАЙПЛАЙН ОБРАБОТКИ
# MAIN PROCESSING PIPELINE
# ==========================================
def run_pipeline():
    """
    Запускает полный пайплайн обработки данных по душевым показателям.
    Runs the complete per capita data processing pipeline.

    Этапы / Steps:
    1. Загрузка Excel-файлов / Load Excel files
    2. Экстракция экономических показателей / Extract economic indicators
    3. Слияние таблиц ВРП / Merge GRP tables
    4. Обработка данных потребительской корзины / Process basket data
    5. Объединение данных / Merge data
    6. Расчет весов / Calculate weights
    7. Геокодирование и фильтрация / Geocoding and filtering
    8. Расчет центров тяжести / Calculate gravity centers
    9. Генерация карты / Generate map
    """
    print("1. Загрузка исходных Excel-файлов (Душевой ВРП)...")
    print("1. Loading source Excel files (Per capita GRP)...")

    # Определяем пути к файлам / Define file paths
    files = {
        'vrp_excel': 'VRP_s1998.xlsx',  # Файл с данными ВРП / GRP data file
        'basket_excel': 'data (1).xls'  # Файл с данными корзины / Basket data file
    }

    # Проверка существования файлов / Check if files exist
    if not os.path.exists(files['vrp_excel']):
        raise FileNotFoundError(f"Ошибка: не найден файл {files['vrp_excel']}")
    if not os.path.exists(files['basket_excel']):
        raise FileNotFoundError(f"Ошибка: не найден файл {files['basket_excel']}")

    # Читаем листы '3' и '4' — это ВРП на душу населения
    # Read sheets '3' and '4' - these are per capita GRP
    try:
        df_vrp_old = pd.read_excel(files['vrp_excel'], sheet_name='3', header=2, engine='openpyxl')
        df_vrp_new = pd.read_excel(files['vrp_excel'], sheet_name='4', header=2, engine='openpyxl')
    except Exception:
        # Альтернативная попытка с индексами листов / Alternative attempt with sheet indices
        df_vrp_old = pd.read_excel(files['vrp_excel'], sheet_name=3, header=2, engine='openpyxl')
        df_vrp_new = pd.read_excel(files['vrp_excel'], sheet_name=4, header=2, engine='openpyxl')

    # Загрузка данных потребительской корзины / Load consumer basket data
    try:
        df_basket_raw = pd.read_excel(files['basket_excel'], sheet_name='Данные', header=2, engine='xlrd')
    except Exception:
        df_basket_raw = pd.read_excel(files['basket_excel'], sheet_name=1, header=2, engine='xlrd')

    df_basket = df_basket_raw.copy()

    print("2. Экстракция экономических показателей (на душу населения)...")
    print("2. Extracting economic indicators (per capita)...")

    # Переименовываем первую колонку в 'Регион' / Rename first column to 'Region'
    df_vrp_old.rename(columns={df_vrp_old.columns[0]: 'Регион'}, inplace=True)
    df_vrp_new.rename(columns={df_vrp_new.columns[0]: 'Регион'}, inplace=True)

    # Приводим названия колонок к строковому типу / Convert column names to string
    df_vrp_old.columns = df_vrp_old.columns.astype(str)
    df_vrp_new.columns = df_vrp_new.columns.astype(str)

    # Динамический поиск колонок с учетом возможных сносок Росстата (типа 20141, 20244)
    # Dynamic search for columns considering possible Rosstat footnotes (e.g., 20141, 20244)
    col_2004 = [c for c in df_vrp_old.columns if '2004' in c][0]
    col_2014 = [c for c in df_vrp_old.columns if '2014' in c][0]
    col_2024 = [c for c in df_vrp_new.columns if '2024' in c][0]

    # Извлекаем и сразу переименовываем в чистые "красивые" года
    # Extract and immediately rename to clean "beautiful" years
    df_vrp_old_extracted = df_vrp_old[['Регион', col_2004, col_2014]].copy()
    df_vrp_old_extracted.rename(columns={col_2004: '2004', col_2014: '2014'}, inplace=True)

    df_vrp_new_extracted = df_vrp_new[['Регион', col_2024]].copy()
    df_vrp_new_extracted.rename(columns={col_2024: '2024'}, inplace=True)

    print("3. Слияние ВРП таблиц...")
    print("3. Merging GRP tables...")

    # Нормализуем названия регионов для слияния / Normalize region names for merging
    df_vrp_old_extracted['Normal_Name'] = df_vrp_old_extracted['Регион'].apply(normalize_region_name)
    df_vrp_new_extracted['Normal_Name'] = df_vrp_new_extracted['Регион'].apply(normalize_region_name)

    # Выполняем слияние / Perform merge
    df_vrp = pd.merge(
        df_vrp_old_extracted[['Normal_Name', '2004', '2014']],
        df_vrp_new_extracted[['Normal_Name', '2024', 'Регион']],
        on='Normal_Name',
        how='inner'
    )

    print("4. Оптимизированная обработка данных потребительской корзины...")
    print("4. Optimized processing of consumer basket data...")

    # Переименовываем первую колонку / Rename first column
    df_basket.rename(columns={df_basket.columns[0]: 'Регион_Корзины'}, inplace=True)
    df_basket.columns = df_basket.columns.astype(str)

    # Нормализуем названия / Normalize names
    normal_names = df_basket['Регион_Корзины'].apply(normalize_region_name)
    new_cols = {'Normal_Name': normal_names}

    # Извлекаем данные по годам из корзины / Extract yearly data from basket
    for year in ['2004', '2014', '2024']:
        # Ищем колонки, содержащие год / Find columns containing the year
        year_cols = [c for c in df_basket.columns if f"{year}." in c or c == year]
        if not year_cols:
            year_cols = [c for c in df_basket.columns if year in c]

        # Обрабатываем найденные колонки / Process found columns
        temp_year_df = df_basket[year_cols].copy()
        for col in year_cols:
            # Заменяем запятые на точки и удаляем пробелы / Replace commas with dots and remove spaces
            temp_year_df[col] = temp_year_df[col].astype(str).str.replace(',', '.').str.replace(r'\s+', '', regex=True)
            temp_year_df[col] = pd.to_numeric(temp_year_df[col], errors='coerce')

        # Вычисляем среднее значение по колонкам (на случай дублирования)
        # Calculate average across columns (in case of duplication)
        new_cols[f'Basket_{year}'] = temp_year_df.mean(axis=1)

    # Создаем объединенный DataFrame / Create merged DataFrame
    df_basket_calculated = pd.concat([df_basket[['Регион_Корзины']], pd.DataFrame(new_cols)], axis=1)

    print("5. Финальное объединение данных...")
    print("5. Final data merging...")

    # Объединяем ВРП и данные корзины / Merge GRP and basket data
    final_df = pd.merge(
        df_vrp,
        df_basket_calculated[['Normal_Name', 'Basket_2004', 'Basket_2014', 'Basket_2024']],
        on='Normal_Name',
        how='inner'
    )

    print("6. Расчет специфических весов (Душевой ВРП / Средняя корзина)...")
    print("6. Calculating specific weights (Per capita GRP / Average basket)...")

    # Рассчитываем весовые коэффициенты для каждого года / Calculate weight coefficients for each year
    for year in ['2004', '2014', '2024']:
        # Очищаем данные ВРП / Clean GRP data
        final_df[year] = final_df[year].astype(str).str.replace(',', '.').str.replace(r'\s+', '', regex=True)
        final_df[year] = pd.to_numeric(final_df[year], errors='coerce')

        # Вес = Душевой ВРП / Стоимость корзины на 1 человека в месяц
        # Weight = Per capita GRP / Monthly basket cost per person
        final_df[f'Weight_{year}'] = final_df[year] / final_df[f'Basket_{year}']

    print("7. Геокодирование и фильтрация...")
    print("7. Geocoding and filtering...")

    # Применяем фильтр Валиулина / Apply Valiulin filter
    final_df = apply_valiulin_filter(final_df, region_col='Регион')

    # Добавляем координаты регионов / Add region coordinates
    final_df['Lat'] = final_df['Normal_Name'].map(lambda x: REGIONAL_COORDINATES.get(x, (None, None))[0])
    final_df['Lon'] = final_df['Normal_Name'].map(lambda x: REGIONAL_COORDINATES.get(x, (None, None))[1])

    # Удаляем строки без координат / Remove rows without coordinates
    final_df = final_df.dropna(subset=['Lat', 'Lon'])

    print(f"   [Инфо]: В финальный расчет вошло {len(final_df)} регионов.")
    print(f"   [Info]: {len(final_df)} regions included in final calculation.")

    print("8. Расчет географических центров тяжести (По новой модели)...")
    print("8. Calculating geographical gravity centers (Using new model)...")

    # Рассчитываем центры тяжести для каждого года / Calculate gravity centers for each year
    lat_2004, lon_2004 = calculate_spherical_gravity_center(final_df, 'Lat', 'Lon', 'Weight_2004')
    lat_2014, lon_2014 = calculate_spherical_gravity_center(final_df, 'Lat', 'Lon', 'Weight_2014')
    lat_2024, lon_2024 = calculate_spherical_gravity_center(final_df, 'Lat', 'Lon', 'Weight_2024')

    # Выводим результаты / Print results
    print(f"   Год 2004 (Душевой) -> Широта: {lat_2004:.4f}, Долгота: {lon_2004:.4f}")
    print(f"   Year 2004 (Per capita) -> Latitude: {lat_2004:.4f}, Longitude: {lon_2004:.4f}")
    print(f"   Год 2014 (Душевой) -> Широта: {lat_2014:.4f}, Долгота: {lon_2014:.4f}")
    print(f"   Year 2014 (Per capita) -> Latitude: {lat_2014:.4f}, Longitude: {lon_2014:.4f}")

    # Проверяем наличие данных для 2024 года / Check if 2024 data exists
    if lat_2024 is not None and lon_2024 is not None:
        print(f"   Год 2024 (Душевой) -> Широта: {lat_2024:.4f}, Долгота: {lon_2024:.4f}")
        print(f"   Year 2024 (Per capita) -> Latitude: {lat_2024:.4f}, Longitude: {lon_2024:.4f}")
    else:
        print("   Год 2024 -> КРИТИЧЕСКАЯ ОШИБКА: Нет данных для расчета!")
        print("   Year 2024 -> CRITICAL ERROR: No data for calculation!")
        return

    print("9. Запуск генерации карты...")
    print("9. Starting map generation...")

    # Создаем карту с центром на Урале / Create map centered on Ural region
    m = folium.Map(location=[54.5, 58.5], zoom_start=5, tiles='OpenStreetMap')

    # Определяем точки для отображения / Define points to display
    points = [
        {"year": "2004 (Душевой)", "coords": [lat_2004, lon_2004], "color": "blue"},
        {"year": "2014 (Душевой)", "coords": [lat_2014, lon_2014], "color": "green"},
        {"year": "2024 (Душевой)", "coords": [lat_2024, lon_2024], "color": "red"}
    ]

    # Добавляем маркеры для каждой точки / Add markers for each point
    for pt in points:
        folium.Marker(
            location=pt["coords"],
            popup=f"<b>Центр тяжести удельного ВРП {pt['year']}</b><br>"
                  f"<b>Per capita GRP gravity center {pt['year']}</b><br>"
                  f"Широта/Latitude: {pt['coords'][0]:.4f}<br>"
                  f"Долгота/Longitude: {pt['coords'][1]:.4f}",
            icon=folium.Icon(color=pt["color"], icon="bookmark")
        ).add_to(m)

    # Добавляем линию, соединяющую точки / Add line connecting the points
    track_coords = [pt["coords"] for pt in points]
    folium.PolyLine(track_coords, color="purple", weight=3, opacity=0.85, dash_array='6, 6').add_to(m)

    # Сохраняем карту / Save the map
    map_output = "per_capita_gravity_centers_2024.html"
    m.save(map_output)
    print(f"✓ Интерактивная карта успешно сохранена как '{map_output}'")
    print(f"✓ Interactive map successfully saved as '{map_output}'")
    print("--- Пайплайн по среднедушевым показателям успешно завершен! ---")
    print("--- Per capita indicators pipeline completed successfully! ---")


if __name__ == "__main__":
    run_pipeline()