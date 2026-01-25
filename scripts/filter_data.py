import sys
import os
import json
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_collection.filters import filter_data_analyst_vacancies, remove_duplicates

def load_vacancies(filename):
  
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_and_save(input_file):

    print("Фильтрация данных аналитиков")
    print(f"Загружаю данные из: {input_file}")
    vacancies = load_vacancies(input_file)
    print(f"Загружено вакансий: {len(vacancies)}")
    

    filtered = filter_data_analyst_vacancies(vacancies)

    filtered = remove_duplicates(filtered)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/processed/analyst_vacancies_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
  
    print(f"\nРезультаты фильтрации:")
    print(f"  Было: {len(vacancies)} вакансий")
    print(f"  Стало: {len(filtered)} аналитических вакансий")
    print(f"\nСохранено в: {output_file}")
    
    return filtered, output_file

if __name__ == "__main__":
    input_file = "data/raw/raw_dataset_20260123_153033.json" 
    
    if not os.path.exists(input_file):
        print(f"Файл не найден: {input_file}")
        print("Сначала выполните: python scripts/collect_data.py")
    else:
        filter_and_save(input_file)