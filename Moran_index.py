"""
Модуль для пространственного анализа экономических показателей регионов РФ.
Module for spatial analysis of economic indicators of Russian regions.

Выполняет расчет глобального и локального индекса Морана для 2004 и 2024 годов,
используя матрицу пространственных весов (соседства).
Calculates global and local Moran's I for 2004 and 2024 using spatial weights matrix (neighborhood).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Настройка шрифтов для корректного отображения кириллицы / Font settings for Cyrillic display
matplotlib.rcParams['font.family'] = ['DejaVu Sans']


# ==========================================
# ФУНКЦИИ НОРМАЛИЗАЦИИ НАЗВАНИЙ РЕГИОНОВ
# FUNCTIONS FOR NORMALIZING REGION NAMES
# ==========================================
def normalize_region_name(name):
    """
    Нормализует название региона для надёжного матчинга.
    Normalizes region name for reliable matching.

    ВАЖНО: порядок проверок критичен — более специфичные проверяются раньше:
    IMPORTANT: order of checks is critical — more specific checks come first:
      - 'сахалин' — до 'саха'/'якути' (иначе Сахалин → Якутия)
      - 'томск'   — до 'омск'        (иначе Томская → Омская)

    Args:
        name (str): Исходное название региона / Original region name

    Returns:
        str: Нормализованное название / Normalized region name
    """
    # Проверка на пустые значения / Check for empty values
    if pd.isna(name) or not isinstance(name, str):
        return ""

    # Приводим к нижнему регистру и удаляем лишние пробелы
    # Convert to lowercase and strip extra spaces
    name = name.lower().strip()

    # 1. Города федерального значения / Federal cities
    if 'моск' in name and 'обл' not in name:
        return 'москва'
    if 'московск' in name:
        return 'московская'
    if 'петерб' in name or 'спб' in name:
        return 'санкт-петербург'
    if 'севастоп' in name:
        return 'севастополь'

    # 2. FIX: Сахалин ДО Саха/Якутии — иначе 'саха' находится в 'сахалинская'
    # FIX: Sakhalin BEFORE Sakha/Yakutia — otherwise 'sakha' is inside 'sakhalinskaya'
    if 'сахалин' in name:
        return 'сахалин'

    if 'крым' in name:
        return 'крым'

    # 3. Слияния регионов (2004 → 2024) / Regional mergers
    if 'перм' in name:
        return 'пермский'
    if 'камчат' in name:
        return 'камчатский'
    if 'забайкал' in name or 'читин' in name or 'чита' in name:
        return 'забайкальский'

    # 4. Переименования и двойные названия / Renamed and double names
    if 'кемеров' in name or 'кузбасс' in name:
        return 'кемеровская'
    if 'ханты' in name or 'хмао' in name or 'югра' in name:
        return 'хмао'
    if 'ямало' in name or 'янао' in name:
        return 'янао'
    if 'чукот' in name:
        return 'чукотский'
    if 'ненец' in name and 'ямало' not in name:
        return 'ненецкий'
    if 'татарстан' in name:
        return 'татарстан'
    if 'башкортостан' in name or 'башкир' in name:
        return 'башкортостан'
    if 'саха' in name or 'якути' in name:
        return 'якутия'
    if 'осети' in name:
        return 'северная осетия'
    if 'чуваш' in name:
        return 'чувашия'
    if 'адыге' in name:
        return 'адыгея'
    if 'тыва' in name or 'тува' in name:
        return 'тыва'

    # Два Алтая и два Новгорода / Two Altai and two Novgorod
    if 'алтайский' in name:
        return 'алтайский'
    if 'алтай' in name:
        return 'алтай'
    if 'нижегород' in name:
        return 'нижегородская'
    if 'новгород' in name:
        return 'новгородская'

    # 5. Остальные республики / Other republics
    if 'кабардин' in name:
        return 'кабардино-балкарская'
    if 'карачаев' in name:
        return 'карачаево-черкесская'
    if 'удмурт' in name:
        return 'удмуртская'
    if 'чечен' in name:
        return 'чечня'
    if 'дагестан' in name:
        return 'дагестан'
    if 'ингуш' in name:
        return 'ингушетия'
    if 'калмык' in name:
        return 'калмыкия'
    if 'карел' in name:
        return 'карелия'
    if 'коми' in name:
        return 'коми'
    if 'марий' in name:
        return 'марий эл'
    if 'мордов' in name:
        return 'мордовия'
    if 'хакас' in name:
        return 'хакасия'
    if 'бурят' in name and 'агин' not in name and 'усть' not in name:
        return 'бурятия'

    # 6. Края и автономная область / Krais and autonomous oblast
    if 'краснодар' in name:
        return 'краснодарский'
    if 'краснояр' in name:
        return 'красноярский'
    if 'примор' in name:
        return 'приморский'
    if 'ставропол' in name:
        return 'ставропольский'
    if 'хабаров' in name:
        return 'хабаровский'
    if 'еврейск' in name:
        return 'еврейская'

    # FIX: Томск ДО Омска — 'омск' является подстрокой 'томск'
    # FIX: Tomsk BEFORE Omsk — 'omsk' is a substring of 'tomsk'
    if 'томск' in name:
        return 'томск'

    # 7. Все остальные области по корню / All other oblasts by root
    obl_names = [
        'амурск', 'архангел', 'астрахан', 'белгород', 'брянск', 'владимир',
        'волгоград', 'вологод', 'воронеж', 'иванов', 'иркутск', 'калининград',
        'калуж', 'киров', 'костром', 'курган', 'курск', 'ленинград', 'липец',
        'магадан', 'мурман', 'новосибир', 'омск', 'оренбург',
        'орлов', 'пензен', 'псков', 'ростов', 'рязан', 'самар', 'саратов',
        'свердлов', 'смолен', 'тамбов', 'тверск', 'тульск',
        'тюмен', 'ульянов', 'челябин', 'ярослав'
    ]
    for obl in obl_names:
        if obl in name:
            return obl

    # Финальная очистка от общих служебных слов / Final cleaning of common stop words
    replacements = ["республика", "область", "край", "г.", "город",
                    "автономный округ", "ао", "округ"]
    for rep in replacements:
        name = name.replace(rep, "")

    # Нормализуем пробелы / Normalize spaces
    return re.sub(r'\s+', ' ', name).strip()


# ==========================================
# ЧТЕНИЕ КОРЗИНЫ (data__1_.xlsx)
# READING BASKET DATA (data__1_.xlsx)
# ==========================================
def read_basket(filepath):
    """
    Читает стоимость потребительской корзины из файла без опоры на header.
    Reads consumer basket cost from file without relying on header.

    Структура файла / File structure:
      Строка 2 (индекс 2): годы — '2004', '2014', '2024' в первых колонках блоков
      Строка 3 (индекс 3): месяцы (январь…декабрь)
      Строки 4+ (индекс 4+): данные по регионам
    2004 занимает колонки 1-12, 2024 — колонки 25-36.

    Args:
        filepath (str): Путь к файлу Excel / Path to Excel file

    Returns:
        pd.DataFrame: DataFrame с колонками Normal_Name, Basket_2004, Basket_2024
    """
    # Загружаем без заголовков для ручной обработки / Load without headers for manual processing
    df = pd.read_excel(filepath, sheet_name='Данные', header=None, engine='openpyxl')

    # Динамически найдём позиции блоков по годам из строки 2
    # Dynamically find block positions by year from row 2
    year_row = df.iloc[2].tolist()
    year_start = {}
    for i, v in enumerate(year_row):
        if str(v) in ['2004', '2024']:
            year_start[str(v)] = i

    # Берем строки с данными (начиная с 5-й строки, индекс 4)
    # Take data rows (starting from row 5, index 4)
    data_rows = df.iloc[4:].copy().reset_index(drop=True)

    # Создаем результирующий DataFrame / Create result DataFrame
    result = pd.DataFrame()
    result['Регион_Корзины'] = data_rows.iloc[:, 0]
    result['Normal_Name'] = result['Регион_Корзины'].apply(normalize_region_name)

    # Обрабатываем каждый год / Process each year
    for year in ['2004', '2024']:
        start = year_start[year]
        # 12 месяцев = 12 столбцов / 12 months = 12 columns
        cols = list(range(start, start + 12))
        block = data_rows.iloc[:, cols].copy()

        # Конвертируем строки в числа / Convert strings to numbers
        for c in block.columns:
            block[c] = pd.to_numeric(
                block[c].astype(str).str.replace(',', '.').str.replace(r'\s+', '', regex=True),
                errors='coerce'
            )

        # Среднее за год / Annual average
        result[f'Basket_{year}'] = block.mean(axis=1)

    return result[['Normal_Name', 'Basket_2004', 'Basket_2024']]


# ==========================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В EXCEL
# SAVING RESULTS TO EXCEL
# ==========================================
def save_results_to_excel(final_df, plot_data, filename="spatial_analysis_results.xlsx"):
    """
    Сохраняет результаты анализа в Excel-файл с двумя листами.
    Saves analysis results to an Excel file with two sheets.

    Args:
        final_df (pd.DataFrame): DataFrame с данными регионов / DataFrame with region data
        plot_data (dict): Словарь с данными для графиков / Dictionary with plot data
        filename (str): Имя выходного файла / Output filename
    """
    print(f"\n6. Сохранение результатов в Excel-файл '{filename}'...")
    print(f"\n6. Saving results to Excel file '{filename}'...")

    export_df = final_df.copy()
    summary_records = []

    # Добавляем колонки для каждого года / Add columns for each year
    for year in ['2004', '2024']:
        z_val = plot_data[year]['z']
        Wz_val = plot_data[year]['Wz']
        local_moran_val = plot_data[year]['local_moran']

        export_df[f'z_std_{year}'] = z_val
        export_df[f'Wz_lag_{year}'] = Wz_val
        export_df[f'Local_Moran_{year}'] = local_moran_val

        # Определяем квадранты для локального индекса Морана
        # Determine quadrants for local Moran's I
        quadrants = []
        for z, wz in zip(z_val, Wz_val):
            if z >= 0 and wz >= 0:
                quadrants.append('High-High (Центр / Ядро)')  # High-High (Core)
            elif z < 0 and wz >= 0:
                quadrants.append('Low-High (Транзитная зона / Аномалия)')  # Low-High (Transition/Anomaly)
            elif z < 0 and wz < 0:
                quadrants.append('Low-Low (Периферия)')  # Low-Low (Periphery)
            else:
                quadrants.append('High-Low (Полюс роста / Одиночка)')  # High-Low (Growth pole/Outlier)
        export_df[f'Quadrant_{year}'] = quadrants

        # Сохраняем глобальные индексы / Save global indices
        summary_records.append({
            'Год исследования': int(year),
            "Глобальный Индекс Морана (Moran's I)": plot_data[year]['beta'],
            'Константа регрессии (Alpha)': plot_data[year]['alpha'],
            'Количество проанализированных субъектов': len(export_df)
        })

    # Формируем порядок колонок / Define column order
    base_cols = ['Регион']
    dynamic_cols = []
    for year in ['2004', '2024']:
        dynamic_cols += [
            f'{year}', f'Basket_{year}', f'Weight_{year}',
            f'z_std_{year}', f'Wz_lag_{year}', f'Local_Moran_{year}', f'Quadrant_{year}'
        ]
    final_cols = base_cols + [c for c in dynamic_cols if c in export_df.columns]
    export_df = export_df[final_cols].sort_values(by='Регион')
    df_summary = pd.DataFrame(summary_records)

    # Сохраняем в Excel / Save to Excel
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            export_df.to_excel(writer, sheet_name='Данные по регионам', index=False)
            df_summary.to_excel(writer, sheet_name='Глобальные индексы', index=False)
        print(f"✓ Результаты успешно сохранены в '{filename}'")
        print(f"✓ Results successfully saved to '{filename}'")
        print(f"  - Лист 'Данные по регионам': {export_df.shape[0]} строк(и)")
        print(f"  - Sheet 'Regional data': {export_df.shape[0]} rows")
        print(f"  - Лист 'Глобальные индексы': сводная динамика Moran's I")
        print(f"  - Sheet 'Global indices': summary of Moran's I dynamics")
    except Exception as e:
        print(f"❌ Ошибка при сохранении в Excel: {e}")
        print(f"❌ Error saving to Excel: {e}")


# ==========================================
# ОСНОВНОЙ ПАЙПЛАЙН
# MAIN PIPELINE
# ==========================================
def run_pipeline():
    """
    Запускает полный пайплайн пространственного анализа.
    Runs the complete spatial analysis pipeline.

    Этапы / Steps:
    1. Загрузка данных ВРП, корзины и матрицы соседства / Load GRP, basket and spatial weights matrix
    2. Нормализация названий регионов / Normalize region names
    3. Объединение таблиц / Merge tables
    4. Построение матрицы пространственных весов / Build spatial weights matrix
    5. Расчет глобального и локального индекса Морана / Calculate global and local Moran's I
    6. Генерация диаграмм рассеяния / Generate Moran scatter plots
    7. Сохранение результатов / Save results to Excel and PNG
    """
    # Определяем пути к файлам / Define file paths
    files = {
        'vrp': 'VRP_s1998.xlsx',
        'basket': 'data (1).xlsx',  # конвертирован из .xls / converted from .xls
        'matrix': 'матрица соседства 2012.xlsx'
    }

    # Проверка наличия файлов / Check if files exist
    for key, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
            raise FileNotFoundError(f"File not found: {path}")

    print("1. Загрузка файлов...")
    print("1. Loading files...")

    # ---- VRP (Валовой региональный продукт) / GRP (Gross Regional Product) ----
    # Лист 3 - ВРП на душу населения 2004, 2014 / Sheet 3 - Per capita GRP 2004, 2014
    # Лист 4 - ВРП на душу населения 2024 / Sheet 4 - Per capita GRP 2024
    df3 = pd.read_excel(files['vrp'], sheet_name='3', header=2, engine='openpyxl')
    df4 = pd.read_excel(files['vrp'], sheet_name='4', header=2, engine='openpyxl')

    # Переименовываем первую колонку / Rename first column
    df3.rename(columns={df3.columns[0]: 'Регион'}, inplace=True)
    df4.rename(columns={df4.columns[0]: 'Регион'}, inplace=True)

    # Приводим названия колонок к строковому типу / Convert column names to string
    df3.columns = df3.columns.astype(str)
    df4.columns = df4.columns.astype(str)

    # Находим колонки с нужными годами / Find columns with required years
    col_2004 = next(c for c in df3.columns if '2004' in c)
    col_2024 = next(c for c in df4.columns if '2024' in c)

    # Извлекаем нужные данные / Extract required data
    vrp3 = df3[['Регион', col_2004]].copy().rename(columns={col_2004: '2004'})
    vrp4 = df4[['Регион', col_2024]].copy().rename(columns={col_2024: '2024'})

    # Нормализуем названия регионов / Normalize region names
    vrp3['Normal_Name'] = vrp3['Регион'].apply(normalize_region_name)
    vrp4['Normal_Name'] = vrp4['Регион'].apply(normalize_region_name)

    # ---- Корзина (потребительская) / Basket (consumer) ----
    print("2. Чтение данных корзины...")
    print("2. Reading basket data...")
    df_basket = read_basket(files['basket'])

    # ---- Матрица соседства / Spatial weights matrix ----
    print("3. Загрузка матрицы соседства...")
    print("3. Loading spatial weights matrix...")
    df_matrix_raw = pd.read_excel(files['matrix'], sheet_name=0, header=2, engine='openpyxl')
    df_matrix_raw.rename(columns={df_matrix_raw.columns[0]: 'Регион_Матрицы'}, inplace=True)
    df_matrix_raw['Normal_Name'] = df_matrix_raw['Регион_Матрицы'].apply(normalize_region_name)
    df_matrix_raw = df_matrix_raw.drop_duplicates(subset=['Normal_Name'])

    # Множество регионов из матрицы — является master-списком
    # Matrix regions set — serves as master list
    matrix_norm_set = set(df_matrix_raw['Normal_Name'].tolist())

    # ---- Объединение VRP 2004 и 2024 / Merging GRP 2004 and 2024 ----
    print("4. Объединение таблиц...")
    print("4. Merging tables...")
    df_vrp = pd.merge(
        vrp3[['Normal_Name', '2004']],
        vrp4[['Normal_Name', '2024', 'Регион']],
        on='Normal_Name', how='inner'
    )

    # Объединяем с корзиной / Merge with basket data
    final_df = pd.merge(df_vrp, df_basket, on='Normal_Name', how='inner')

    # Фильтруем строго по регионам из матрицы (79 субъектов)
    # Filter strictly by regions from matrix (79 subjects)
    final_df = final_df[final_df['Normal_Name'].isin(matrix_norm_set)].copy()

    # Конвертируем ВРП в числа и считаем удельный вес
    # Convert GRP to numbers and calculate specific weight
    for year in ['2004', '2024']:
        final_df[year] = pd.to_numeric(
            final_df[year].astype(str).str.replace(',', '.').str.replace(r'\s+', '', regex=True),
            errors='coerce'
        )
        # Вес = ВРП на душу / стоимость корзины / Weight = Per capita GRP / Basket cost
        final_df[f'Weight_{year}'] = final_df[year] / final_df[f'Basket_{year}']

    # Удаляем строки с пропущенными весами и дубликаты
    # Remove rows with missing weights and duplicates
    final_df = final_df.dropna(subset=['Weight_2004', 'Weight_2024'])
    final_df = final_df.drop_duplicates(subset=['Normal_Name'])

    print(f"   ✓ Итоговая выборка: {len(final_df)} регионов (ожидается 79)")
    print(f"   ✓ Final sample: {len(final_df)} regions (expected 79)")

    # ---- Выравнивание с матрицей соседства / Alignment with spatial weights matrix ----
    print("5. Построение пространственной матрицы W...")
    print("5. Building spatial weights matrix W...")

    # Получаем список колонок матрицы (регионы-соседи) / Get matrix columns (neighbor regions)
    orig_cols = [c for c in df_matrix_raw.columns if c not in ['Регион_Матрицы', 'Normal_Name']]
    norm_col_mapping = {normalize_region_name(c): c for c in orig_cols}

    # Определяем пересечение регионов / Determine intersection of regions
    valid_regions = [r for r in df_matrix_raw['Normal_Name'].tolist()
                     if r in set(final_df['Normal_Name'].tolist())]

    # Выравниваем DataFrame'ы / Align DataFrames
    final_df = (final_df[final_df['Normal_Name'].isin(valid_regions)]
                .set_index('Normal_Name').reindex(valid_regions).reset_index())
    df_matrix_filtered = (df_matrix_raw[df_matrix_raw['Normal_Name'].isin(valid_regions)]
                          .set_index('Normal_Name').reindex(valid_regions).reset_index())

    # Строим матрицу весов / Build weights matrix
    N = len(valid_regions)
    W = np.zeros((N, N))

    for i, r_name in enumerate(valid_regions):
        row_data = df_matrix_filtered.iloc[i]
        for j, c_name in enumerate(valid_regions):
            orig_col = norm_col_mapping.get(c_name)
            if orig_col and orig_col in df_matrix_filtered.columns:
                val = row_data[orig_col]
                if pd.notna(val):
                    W[i, j] = pd.to_numeric(
                        str(val).replace(',', '.').strip(), errors='coerce'
                    ) or 0.0

    # Нормирование строк (row-standardization) / Row normalization
    for i in range(N):
        row_sum = W[i, :].sum()
        if row_sum > 0:
            W[i, :] /= row_sum

    print(f"   ✓ Матрица W размером {N}×{N} построена.")
    print(f"   ✓ Matrix W of size {N}×{N} built.")

    # ---- Индекс Морана / Moran's I calculation ----
    print("6. Расчёт индекса Морана...")
    print("6. Calculating Moran's I...")
    plot_data = {}

    for year in ['2004', '2024']:
        x = final_df[f'Weight_{year}'].to_numpy()
        # Стандартизация / Standardization
        z_std = (x - x.mean()) / x.std()
        # Пространственный лаг / Spatial lag
        Wz = W @ z_std
        # Локальный индекс Морана / Local Moran's I
        local_moran = z_std * Wz
        # Глобальный индекс Морана (коэффициент регрессии) / Global Moran's I (regression coefficient)
        beta, alpha = np.polyfit(z_std, Wz, 1)
        plot_data[year] = {
            'z': z_std, 'Wz': Wz,
            'local_moran': local_moran,
            'beta': beta, 'alpha': alpha,
            'regions': final_df['Регион'].tolist()
        }
        print(f"   Год {year}: Moran's I = {beta:.4f}")
        print(f"   Year {year}: Moran's I = {beta:.4f}")

    # ---- Графики / Plots ----
    print("7. Генерация диаграмм рассеяния Морана...")
    print("7. Generating Moran scatter plots...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    colors = {'2004': '#1f77b4', '2024': '#d62728'}

    for idx, year in enumerate(['2004', '2024']):
        ax = axes[idx]
        z_val = plot_data[year]['z']
        Wz_val = plot_data[year]['Wz']
        beta = plot_data[year]['beta']
        alpha = plot_data[year]['alpha']
        reg_names = plot_data[year]['regions']

        # Точки (регионы) / Points (regions)
        ax.scatter(z_val, Wz_val, color=colors[year], alpha=0.7,
                   edgecolors='k', s=50, label='Субъекты РФ / Russian regions')

        # Линия регрессии / Regression line
        x_range = np.linspace(z_val.min() - 0.5, z_val.max() + 0.5, 100)
        ax.plot(x_range, beta * x_range + alpha, color='black',
                linestyle='--', linewidth=2,
                label=f"Линия регрессии / Regression line (Moran's I = {beta:.3f})")

        # Нулевые линии / Zero lines
        ax.axhline(0, color='black', linestyle='-', alpha=0.2)
        ax.axvline(0, color='black', linestyle='-', alpha=0.2)

        # Подписи квадрантов / Quadrant labels
        ax.text(z_val.max() * 0.3, Wz_val.max() * 0.8,
                'High-High\n(Кластер / Cluster)', fontsize=9, color='darkgreen', weight='bold')
        ax.text(z_val.min() * 0.6, Wz_val.max() * 0.8,
                'Low-High\n(Аномалия / Anomaly)', fontsize=9, color='darkorange', weight='bold')
        ax.text(z_val.min() * 0.6, Wz_val.min() * 0.8,
                'Low-Low\n(Кластер / Cluster)', fontsize=9, color='darkblue', weight='bold')
        ax.text(z_val.max() * 0.3, Wz_val.min() * 0.8,
                'High-Low\n(Аномалия / Anomaly)', fontsize=9, color='purple', weight='bold')

        # Подписи регионов (только для удаленных от центра) / Region labels (only for outliers)
        for i, txt in enumerate(reg_names):
            short = (txt.replace('область', 'обл.')
                     .replace('Республика', 'Респ.')
                     .replace('автономный округ', 'АО'))
            if abs(z_val[i]) > 1 or abs(Wz_val[i]) > 1:
                ax.annotate(short, (z_val[i], Wz_val[i]), fontsize=7.5, alpha=0.9,
                            xytext=(4, 4), textcoords='offset points', weight='semibold')

        # Заголовки и подписи осей / Titles and axis labels
        ax.set_title(f'Диаграмма рассеяния Морана — {year} г.\nMoran Scatter Plot — {year}',
                     fontsize=13, weight='bold')
        ax.set_xlabel('Стандартизированный уровень благосостояния ($z$)\nStandardized welfare level ($z$)',
                      fontsize=11)
        ax.set_ylabel('Пространственный лаг ($Wz$)\nSpatial lag ($Wz$)', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    out_img = 'moran_subplots_2004_2024.png'
    plt.savefig(out_img, dpi=300)
    print(f"✓ Графики сохранены в '{out_img}'")
    print(f"✓ Plots saved to '{out_img}'")

    # ---- Сохранение Excel / Save to Excel ----
    out_xlsx = 'spatial_analysis_results.xlsx'
    save_results_to_excel(final_df, plot_data, filename=out_xlsx)

    plt.show()
    return final_df, plot_data


if __name__ == "__main__":
    run_pipeline()