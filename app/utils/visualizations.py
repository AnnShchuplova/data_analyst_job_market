"""
Утилита для генерации EDA-визуализаций.

Согласно ЧТЗ: «Автоматическая генерация графиков: 
распределение зарплат, топ-10 навыков, распределение вакансий по городам»
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging

logger = logging.getLogger(__name__)

# Настройка шрифтов для русского текста
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SimHei.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_eda_charts(df: pd.DataFrame, output_dir: str = None) -> list:
    """Генерация стандартных EDA-графиков.
    
    Согласно ЧТЗ: набор графиков EDA в формате .png
    
    Parameters:
    -----------
    df : DataFrame с очищенными данными
    output_dir : директория для сохранения графиков
    
    Returns:
    --------
    list : пути к сохранённым файлам
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "reports", "eda"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # 1. Распределение зарплат
    if 'salary_avg' in df.columns:
        fig_path = os.path.join(output_dir, "salary_distribution.png")
        _plot_salary_distribution(df, fig_path)
        saved_files.append(fig_path)
    
    # 2. Топ-10 навыков
    if 'skills_list' in df.columns:
        fig_path = os.path.join(output_dir, "top_skills.png")
        _plot_top_skills(df, fig_path)
        saved_files.append(fig_path)
    
    # 3. Распределение по городам
    area_col = 'area_name' if 'area_name' in df.columns else None
    if area_col:
        fig_path = os.path.join(output_dir, "region_distribution.png")
        _plot_region_distribution(df, area_col, fig_path)
        saved_files.append(fig_path)
    
    # 4. Распределение по опыту
    if 'experience_name' in df.columns:
        fig_path = os.path.join(output_dir, "experience_distribution.png")
        _plot_experience_distribution(df, fig_path)
        saved_files.append(fig_path)
    
    # 5. Распределение по формату работы
    if 'work_format_name' in df.columns:
        fig_path = os.path.join(output_dir, "work_format_distribution.png")
        _plot_work_format_distribution(df, fig_path)
        saved_files.append(fig_path)
    
    logger.info(f"Сгенерировано {len(saved_files)} EDA-графиков в {output_dir}")
    plt.close('all')
    
    return saved_files


def _plot_salary_distribution(df, filepath):
    """Распределение зарплат."""
    salary_data = df['salary_avg'].dropna()
    salary_data = salary_data[salary_data > 0]
    
    if len(salary_data) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Гистограмма
    axes[0].hist(salary_data / 1000, bins=30, color='#4C72B0', edgecolor='white', alpha=0.8)
    axes[0].axvline(salary_data.mean() / 1000, color='red', linestyle='--', label=f'Среднее: {salary_data.mean()/1000:.0f}K')
    axes[0].axvline(salary_data.median() / 1000, color='orange', linestyle='--', label=f'Медиана: {salary_data.median()/1000:.0f}K')
    axes[0].set_xlabel('Зарплата (тыс. руб.)', fontsize=11)
    axes[0].set_ylabel('Количество вакансий', fontsize=11)
    axes[0].set_title('Распределение зарплат', fontsize=13)
    axes[0].legend(loc='best', fontsize=9)
    
    # Box plot по опыту
    if 'experience_name' in df.columns:
        df_salary = df.dropna(subset=['salary_avg'])
        df_salary = df_salary[df_salary['salary_avg'] > 0]
        exp_order = df_salary.groupby('experience_name')['salary_avg'].median().sort_values().index
        if len(exp_order) > 0:
            import seaborn as sns
            sns.boxplot(data=df_salary, x='experience_name', y='salary_avg', order=exp_order, ax=axes[1])
            axes[1].set_ylabel('Зарплата (руб.)', fontsize=11)
            axes[1].set_xlabel('')
            axes[1].set_title('Зарплаты по опыту', fontsize=13)
            axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Сохранено: {filepath}")


def _plot_top_skills(df, filepath):
    """Топ-10 востребованных навыков."""
    import ast
    from collections import Counter
    
    all_skills = []
    for val in df['skills_list'].dropna():
        if isinstance(val, list):
            all_skills.extend(val)
        elif isinstance(val, str) and val.strip():
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    all_skills.extend(parsed)
            except (ValueError, SyntaxError):
                pass
    
    if not all_skills:
        return
    
    top_skills = Counter(all_skills).most_common(15)
    skills, counts = zip(*top_skills)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(skills)), counts, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(skills))))
    ax.set_yticks(range(len(skills)))
    ax.set_yticklabels(skills)
    ax.invert_yaxis()
    ax.set_xlabel('Количество вакансий', fontsize=11)
    ax.set_title('Топ-15 востребованных навыков', fontsize=13)
    
    # Добавляем значения на столбцы
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Сохранено: {filepath}")


def _plot_region_distribution(df, area_col, filepath):
    """Распределение вакансий по городам."""
    region_counts = df[area_col].fillna('Не указан').value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(region_counts)), region_counts.values, 
                   color=plt.cm.Set2(np.linspace(0, 1, len(region_counts))))
    ax.set_yticks(range(len(region_counts)))
    ax.set_yticklabels(region_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel('Количество вакансий', fontsize=11)
    ax.set_title('Топ-10 городов по количеству вакансий', fontsize=13)
    
    for bar, count in zip(bars, region_counts.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Сохранено: {filepath}")


def _plot_experience_distribution(df, filepath):
    """Распределение по опыту работы."""
    exp_counts = df['experience_name'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    ax.pie(exp_counts.values, labels=exp_counts.index, autopct='%1.1f%%',
           colors=colors[:len(exp_counts)], startangle=90)
    ax.set_title('Распределение по опыту работы', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Сохранено: {filepath}")


def _plot_work_format_distribution(df, filepath):
    """Распределение по формату работы."""
    format_counts = df['work_format_name'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(format_counts)), format_counts.values, 
                  color=plt.cm.Paired(np.linspace(0, 1, len(format_counts))))
    ax.set_xticks(range(len(format_counts)))
    ax.set_xticklabels(format_counts.index, rotation=15)
    ax.set_ylabel('Количество вакансий', fontsize=11)
    ax.set_title('Распределение по формату работы', fontsize=13)
    
    for bar, count in zip(bars, format_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Сохранено: {filepath}")
