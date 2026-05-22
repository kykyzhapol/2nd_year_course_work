"""
Модуль для расчета географических центров тяжести экономической активности регионов РФ.
Module for calculating geographical gravity centers of economic activity in Russian regions.

Загружает данные ВРП и потребительской корзины, вычисляет веса регионов,
и определяет центры тяжести на сфере для 2004, 2014 и 2024 годов.
Loads GRP and consumer basket data, calculates regional weights,
and determines spherical gravity centers for 2004, 2014, and 2024.
"""

import os
import re
import numpy as np
import pandas as pd
import folium
from math import radians, cos, sin, atan2, sqrt, degrees

# ==========================================
# ПОЛНЫЙ СПРАВОЧНИК КООРДИНАТ РЕГИОНОВ РФ (82 СУБЪЕКТА)
# COMPLETE DIRECTORY OF RUSSIAN REGION COORDINATES (82 SUBJECTS)
# ==========================================
REGIONAL_COORDINATES = {
    # Центральный ФО / Central Federal District
    'белгородская': (50.5997, 36.5982), 'брянская': (52.8876, 33.4058), 'владимирская': (56.0416, 40.4431),
    'воронежская': (51.6720, 39.1843), 'ивановская': (56.9972, 41.6143), 'калужская': (54.5293, 36.2754),
    'костромская': (57.7665, 40.9269), 'курская': (51.7156, 36.1926), 'липецкая': (52.6031, 39.5708),
    'московская': (55.7115, 37.8920), 'орловская': (52.9651, 36.0785), 'рязанская': (54.6095, 39.7126),
    'смоленская': (54.7818, 32.0401), 'тамбовская': (52.7317, 41.4433), 'тверская': (56.9942, 34.9048),
    'тульская': (54.1961, 37.6182), 'ярославская': (57.7314, 39.0534), 'москва': (55.7558, 37.6173),

    # Северо-Западный ФО / Northwestern Federal District
    'карелия': (61.7849, 34.3469), 'коми': (61.6688, 50.8365), 'архангельская': (64.5472, 40.5601),
    'вологодская': (59.2205, 39.8915), 'калининградская': (54.7671, 21.4682), 'ленинградская': (59.9645, 31.0267),
    'мурманская': (68.9585, 33.0827), 'новгородская': (58.5256, 31.2742), 'псковская': (57.8136, 28.3496),
    'санкт-петербург': (59.9343, 30.3351), 'ненецкий': (67.6380, 53.0070),

    # Южный ФО / Southern Federal District
    'адыгея': (44.6098, 40.1006), 'калмыкия': (46.3078, 44.2558), 'краснодарский': (45.0355, 38.9753),
    'астраханская': (46.3476, 48.0302), 'волгоградская': (48.7071, 44.5170), 'ростовская': (47.2364, 39.7139),

    # Северо-Кавказский ФО / North Caucasian Federal District
    'дагестан': (42.9831, 47.5047), 'ингушетия': (43.1669, 44.8118), 'кабардино-балкарская': (43.4834, 43.6071),
    'карачаево-черкесская': (44.2233, 42.0578), 'северная осетия': (43.0367, 44.6678),
    'ставропольский': (45.0445, 41.9690),

    # Приволжский ФО / Volga Federal District
    'башкортостан': (54.4078, 56.0271), 'марий эл': (56.6343, 47.8998), 'мордовия': (54.1838, 45.1749),
    'татарстан': (55.4372, 50.5369), 'удмуртская': (56.8498, 53.2045), 'чувашия': (55.4357, 47.1246),
    'пермский': (58.0105, 56.2502), 'кировская': (58.5966, 49.6547), 'нижегородская': (56.2464, 44.5700),
    'оренбургская': (51.7666, 55.0988), 'пензенская': (53.1959, 45.0183), 'самарская': (53.2001, 50.1500),
    'саратовская': (51.5410, 46.0086), 'ульяновская': (54.3169, 48.4052),

    # Уральский ФО / Ural Federal District
    'курганская': (55.4507, 65.3333), 'свердловская': (58.4503, 61.2720), 'тюменская': (57.1522, 65.5272),
    'челябинская': (54.7247, 61.0242), 'ханты-мансийский': (61.0042, 69.0019), 'ямало-ненецкий': (66.5300, 70.3200),

    # Сибирский ФО / Siberian Federal District
    'алтай': (51.5167, 86.0000), 'тыва': (51.7147, 94.4534), 'хакасия': (53.7156, 91.4292),
    'алтайский': (52.5000, 83.0000), 'красноярский': (64.2132, 98.4870), 'иркутская': (56.4914, 103.5222),
    'кемеровская': (54.3415, 86.2699), 'новосибирская': (55.2750, 79.9114), 'омская': (54.9885, 73.3242),
    'томская': (56.4847, 84.9482),

    # Дальневосточный ФО / Far Eastern Federal District
    'бурятия': (53.1549, 109.8315), 'якутия': (66.3817, 124.9788), 'забайкальский': (52.2033, 116.5167),
    'камчатский': (56.5100, 159.2600), 'приморский': (45.1664, 134.1378), 'хабаровский': (54.5126, 136.0028),
    'амурская': (53.6667, 127.8333), 'магаданская': (62.8837, 151.7820), 'сахалинская': (50.6083, 142.7483),
    'еврейская': (48.3644, 132.8222), 'чукотский': (66.2573, 172.4332)
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
    Применяет фильтр Валиулина для исключения проблемных регионов.
    Applies Valiulin filter to exclude problematic regions.

    Фильтр исключает: автономные округа (кроме Чукотки, ХМАО, ЯНАО),
    исключенные автором регионы, технические строки.
    Filter excludes: small autonomous districts (except Chukotka, KhMAO, YaNAO),
    excluded by author regions, technical rows.

    Args:
        df (pd.DataFrame): DataFrame с данными / DataFrame with data
        region_col (str): Название колонки с регионами / Region column name

    Returns:
        pd.DataFrame: Отфильтрованный DataFrame / Filtered DataFrame
    """

    def is_valid(name):
        # Проверка на пустые значения / Check for empty values
        if pd.isna(name) or not isinstance(name, str):
            return False

        n_lower = name.lower().strip()

        # Фильтр Валиулина: исключаем АО
        # Valiulin filter: exclude autonomous districts
        if 'автономный округ' in n_lower or ' ао' in n_lower or n_lower.endswith(' ао'):
            if 'чукотский' not in n_lower and 'ханты' not in n_lower and 'ямало' not in n_lower:
                return False

        exclusions = [
            'крым', 'севастополь', 'чеченск', 'чечня',
            'донецк', 'луганск', 'запорож', 'херсон', 'днр', 'лнр'
        ]
        if any(excl in n_lower for excl in exclusions):
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
    Запускает полный пайплайн обработки данных.
    Runs the complete data processing pipeline.

    Этапы / Steps:
    1. Загрузка Excel-файлов / Load Excel files
    2. Экстракция экономических показателей / Extract economic indicators
    3. Слияние таблиц ВРП / Merge GRP tables
    4. Сопоставление названий регионов / Match region names
    5. Объединение данных / Merge data
    6. Расчет весов / Calculate weights
    7. Геокодирование и фильтрация / Geocoding and filtering
    8. Расчет центров тяжести / Calculate gravity centers
    9. Генерация карты / Generate map
    """
    print("1. Загрузка исходных Excel-файлов...")
    print("1. Loading source Excel files...")

    # Определяем пути к файлам / Define file paths
    files = {
        'vrp_excel': 'VRP_s1998.xlsx',  # Файл с данными ВРП / GRP data file
        'basket_excel': 'data (1).xls'  # Файл с данными корзины / Basket data file
    }

    # Проверка существования файлов / Check if files exist
    if not os.path.exists(files['vrp_excel']) or not os.path.exists(files['basket_excel']):
        raise FileNotFoundError("Критическая ошибка: Проверьте наличие файлов VRP_s1998.xlsx и data (1).xls")
        raise FileNotFoundError("Critical error: Check that VRP_s1998.xlsx and data (1).xls exist")

    # Загрузка данных ВРП (старые и новые показатели) / Load GRP data (old and new indicators)
    try:
        df_vrp_old = pd.read_excel(files['vrp_excel'], sheet_name='1', header=2, engine='openpyxl')
        df_vrp_new = pd.read_excel(files['vrp_excel'], sheet_name='2', header=2, engine='openpyxl')
    except Exception:
        # Альтернативная попытка с индексами листов / Alternative attempt with sheet indices
        df_vrp_old = pd.read_excel(files['vrp_excel'], sheet_name=1, header=2, engine='openpyxl')
        df_vrp_new = pd.read_excel(files['vrp_excel'], sheet_name=2, header=2, engine='openpyxl')

    # Загрузка данных потребительской корзины / Load consumer basket data
    try:
        df_basket_raw = pd.read_excel(files['basket_excel'], sheet_name='Данные', header=2, engine='xlrd')
    except Exception:
        df_basket_raw = pd.read_excel(files['basket_excel'], sheet_name=1, header=2, engine='xlrd')

    df_basket = df_basket_raw.copy()

    print("2. Экстракция экономических показателей...")
    print("2. Extracting economic indicators...")

    # Переименовываем первую колонку в 'Регион' / Rename first column to 'Region'
    df_vrp_old.rename(columns={df_vrp_old.columns[0]: 'Регион'}, inplace=True)
    df_vrp_new.rename(columns={df_vrp_new.columns[0]: 'Регион'}, inplace=True)

    # Приводим названия колонок к строковому типу / Convert column names to string
    df_vrp_old.columns = df_vrp_old.columns.astype(str)
    df_vrp_new.columns = df_vrp_new.columns.astype(str)

    # Находим колонку с 2024 годом / Find column for year 2024
    col_2024 = [c for c in df_vrp_new.columns if '2024' in c][0]

    # Извлекаем нужные колонки / Extract required columns
    df_vrp_old_extracted = df_vrp_old[['Регион', '2004', '2014']].copy()
    df_vrp_new_extracted = df_vrp_new[['Регион', col_2024]].copy().rename(columns={col_2024: '2024'})

    print("3. Слияние ВРП таблиц...")
    print("3. Merging GRP tables...")

    # Нормализуем названия регионов для слияния / Normalize region names for merging
    df_vrp_old_extracted['Normal_Name'] = df_vrp_old_extracted['Регион'].apply(normalize_region_name)
    df_vrp_new_extracted['Normal_Name'] = df_vrp_new_extracted['Регион'].apply(normalize_region_name)

    # Выполняем слияние / Perform merge
    df_vrp = pd.merge(
        df_vrp_old_extracted[['Normal_Name', '2004', '2014']],
        df_vrp_new_extracted[['Normal_Name', '2024', 'Регион']],
        on='Normal_Name', how='inner'
    )

    print("4. Интеллектуальное сопоставление названий регионов корзины...")
    print("4. Intelligent matching of basket region names...")

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

    # Создаем DataFrame с вычисленными значениями / Create DataFrame with calculated values
    df_basket_calculated = pd.DataFrame(new_cols)

    print("5. Финальное объединение данных...")
    print("5. Final data merging...")

    # Объединяем ВРП и данные корзины / Merge GRP and basket data
    final_df = pd.merge(
        df_vrp,
        df_basket_calculated[['Normal_Name', 'Basket_2004', 'Basket_2014', 'Basket_2024']],
        on='Normal_Name', how='inner'
    )

    print("6. Расчет специфических весов...")
    print("6. Calculating specific weights...")

    # Рассчитываем весовые коэффициенты для каждого года / Calculate weight coefficients for each year
    for year in ['2004', '2014', '2024']:
        # Очищаем данные ВРП / Clean GRP data
        final_df[year] = final_df[year].astype(str).str.replace(',', '.').str.replace(r'\s+', '', regex=True)
        final_df[year] = pd.to_numeric(final_df[year], errors='coerce')

        # Вес = ВРП региона / корзина региона / Weight = Regional GRP / Regional basket
        final_df[f'Weight_{year}'] = final_df[year] / final_df[f'Basket_{year}']

    print("7. Геокодирование и фильтрация...")
    print("7. Geocoding and filtering...")

    # Применяем фильтр Валиулина / Apply Valiulin filter
    final_df = apply_valiulin_filter(final_df, region_col='Регион')

    # Добавляем координаты регионов / Add region coordinates
    final_df['Lat'] = final_df['Normal_Name'].map(lambda x: REGIONAL_COORDINATES.get(x, (None, None))[0])
    final_df['Lon'] = final_df['Normal_Name'].map(lambda x: REGIONAL_COORDINATES.get(x, (None, None))[1])

    # Удаляем строки без координат и дубликаты / Remove rows without coordinates and duplicates
    final_df = final_df.dropna(subset=['Lat', 'Lon']).drop_duplicates(subset=['Normal_Name'])

    print(f"   [Инфо]: В финальный расчет вошло {len(final_df)} уникальных субъектов РФ.")
    print(f"   [Info]: {len(final_df)} unique Russian subjects included in final calculation.")

    print("8. Расчет географических центров тяжести для всех периодов...")
    print("8. Calculating geographical gravity centers for all periods...")

    # Рассчитываем центры тяжести для каждого года / Calculate gravity centers for each year
    lat_2004, lon_2004 = calculate_spherical_gravity_center(final_df, 'Lat', 'Lon', 'Weight_2004')
    lat_2014, lon_2014 = calculate_spherical_gravity_center(final_df, 'Lat', 'Lon', 'Weight_2014')
    lat_2024, lon_2024 = calculate_spherical_gravity_center(final_df, 'Lat', 'Lon', 'Weight_2024')

    # Выводим результаты / Print results
    print(f"   Год 2004 -> Широта: {lat_2004:.4f}, Долгота: {lon_2004:.4f}")
    print(f"   Year 2004 -> Latitude: {lat_2004:.4f}, Longitude: {lon_2004:.4f}")
    print(f"   Год 2014 -> Широта: {lat_2014:.4f}, Долгота: {lon_2014:.4f}")
    print(f"   Year 2014 -> Latitude: {lat_2014:.4f}, Longitude: {lon_2014:.4f}")
    print(f"   Год 2024 -> Широта: {lat_2024:.4f}, Долгота: {lon_2024:.4f}")
    print(f"   Year 2024 -> Latitude: {lat_2024:.4f}, Longitude: {lon_2024:.4f}")

    print("9. Запуск генерации карты...")
    print("9. Starting map generation...")

    # Создаем карту с центром около Урала/Сибири (Уфа/Челябинск)
    # Create map centered near Ural/Siberia (Ufa/Chelyabinsk)
    m = folium.Map(location=[55.0, 60.0], zoom_start=5, tiles='OpenStreetMap')

    # Определяем точки для отображения / Define points to display
    points = [
        {"year": "2004", "coords": [lat_2004, lon_2004], "color": "blue"},
        {"year": "2014", "coords": [lat_2014, lon_2014], "color": "green"},
        {"year": "2024", "coords": [lat_2024, lon_2024], "color": "red"}
    ]

    # Добавляем маркеры для каждой точки / Add markers for each point
    for pt in points:
        folium.Marker(
            location=pt["coords"],
            popup=f"<b>Центр тяжести экономики {pt['year']}</b><br>"
                  f"<b>Economic gravity center {pt['year']}</b><br>"
                  f"Широта/Latitude: {pt['coords'][0]:.4f}<br>"
                  f"Долгота/Longitude: {pt['coords'][1]:.4f}",
            icon=folium.Icon(color=pt["color"], icon="info-sign")
        ).add_to(m)

    # Добавляем линию, соединяющую точки / Add line connecting the points
    track_coords = [pt["coords"] for pt in points]
    folium.PolyLine(track_coords, color="black", weight=2.5, opacity=0.8, dash_array='5, 5').add_to(m)

    # Сохраняем карту / Save the map
    map_output = "economic_gravity_centers_2024.html"
    m.save(map_output)
    print(f"✓ Карта успешно сохранена как '{map_output}'")
    print(f"✓ Map successfully saved as '{map_output}'")
    print("--- Выполнение завершено успешно! ---")
    print("--- Execution completed successfully! ---")


if __name__ == "__main__":
    run_pipeline()