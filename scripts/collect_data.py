import sys
import os
import time
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from datetime import datetime
from src.data_collection.hh_parser import HHParser

def collect_raw_data(pages_per_role=3, total_pages_limit=200):

    parser = HHParser(delay=0.3)
    
    ANALYTIC_ROLES = [
        ("156", "BI-аналитик, аналитик данных", "аналитик данных"),
        ("10", "Аналитик", "аналитик"),
        ("150", "Бизнес-аналитик", "бизнес аналитик"),
        ("134", "Финансовый аналитик", "финансовый аналитик"),
        ("164", "Продуктовый аналитик", "продуктовый аналитик"),
        ("148", "Системный аналитик", "системный аналитик"),
        ("163", "Маркетолог-аналитик", "маркетинг аналитик"),
        ("157", "Руководитель отдела аналитики", "руководитель аналитики"),
    ]
    
    all_vacancies = []
    total_requests = 0
    
    for role_id, role_name, query in ANALYTIC_ROLES:
        print(f"Роль: {role_name} (ID: {role_id})")
        pages_to_fetch = min(pages_per_role, total_pages_limit)
        
        if total_requests + pages_to_fetch > total_pages_limit:
            pages_to_fetch = total_pages_limit - total_requests
        
        try:
            role_vacancies = parser.fetch_vacancies(
                query=query,
                pages=pages_to_fetch,
                professional_role=role_id, 
                only_with_salary=False
            )
            
            existing_ids = {v['id'] for v in all_vacancies}
            new_vacancies = [v for v in role_vacancies if v['id'] not in existing_ids]
            
            all_vacancies.extend(new_vacancies)
            total_requests += pages_to_fetch
            
            print(f"Собрано {len(new_vacancies)} вакансий")

            if len(ANALYTIC_ROLES) > 1:
                time.sleep(1)
            
        except Exception as e:
            print(f"Ошибка для роли {role_id}: {e}")
            continue
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/raw_dataset_{timestamp}.json"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_vacancies, f, ensure_ascii=False, indent=2)
    
    print(f"Собрано уникальных вакансий: {len(all_vacancies)}")
    print(f"Файл c данными {filename}")

    
    return all_vacancies, filename


if __name__ == "__main__":
    vacancies, filename = collect_raw_data(pages_per_role=5, total_pages_limit=30)
    