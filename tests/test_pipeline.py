import sys
import os
import time
from datetime import datetime
import traceback
    
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from scripts.collect_data import collect_raw_data
from scripts.filter_data import filter_and_save
from src.data_processing.cleaner import DataCleaner
    

def test_full_pipeline(pages=10):

    print("Тестирование сбора данных")
    print()
    
    total_start = time.time()
    print("Сбор данных с HH.ru")
    
    try:
        start = time.time()
        raw_vacancies, raw_filename = collect_raw_data(pages_per_role=min(10, pages))
        collection_time = time.time() - start
        
        print(f"Собрано: {len(raw_vacancies)} вакансий")
        print(f"Время сбора: {collection_time:.1f} сек")
        print(f"Файл: {raw_filename}")
        
        if len(raw_vacancies) == 0:
            print("Не собрано ни одной вакансии")
            return False
            
    except Exception as e:
        print(f"Ошибка сбора данных: {e}")
        traceback.print_exc()
        return False
    
    print()
   
    print("Фильтрация вакансий")
    
    try:
        
        start = time.time()
        filtered_vacancies, filtered_filename = filter_and_save(raw_filename)
        filter_time = time.time() - start
        
        print(f"Отфильтровано {len(filtered_vacancies)} вакансий аналитиков")
        print(f"{filter_time:.1f} сек")
        print(f"Файл: {filtered_filename}")
        
        if len(filtered_vacancies) == 0:
            print("после фильтрации не осталось вакансий")
            
    except Exception as e:
        print(f"Ошибка на фильтрации: {e}")
        traceback.print_exc()
        return False
    
    print()
    
    print("Очистка данных")
    
    try:
        cleaner = DataCleaner()
        start = time.time()
        cleaned_df = cleaner.run_full_clean(filtered_filename)
        clean_time = time.time() - start
        
        print(f"Очищено {len(cleaned_df)} строк")
            
    except Exception as e:
        print(f"Ошибка очистки: {e}")
        traceback.print_exc()
    
    print()
    
    total_time = time.time() - total_start
    
    print(f"Общее время {total_time:.1f} сек")
   
    if 'cleaned_df' in locals():
        print(f"Очищенных строк: {len(cleaned_df)}")
    
    print()

    save_test_report(raw_vacancies, filtered_vacancies, total_time)
    
    return True

def save_test_report(raw_vacancies, filtered_vacancies, total_time):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"reports/test_report_{timestamp}.txt"
        
        os.makedirs("reports", exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Ответ о тесте пайплайна")
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            f.write(f"Сырых вакансий: {len(raw_vacancies)}\n")
            f.write(f"После фильтрации: {len(filtered_vacancies)}\n")
            
            f.write(f"Общее время: {total_time:.1f} сек\n\n")
            
            f.write("Примеры отфильтрованных вакансий:\n")
            for i, vac in enumerate(filtered_vacancies[:10], 1):
                name = vac.get('name', 'Без названия')
                salary = vac.get('salary', {})
                salary_str = f"{salary.get('from', '')}-{salary.get('to', '')}" if salary else "нет"
                f.write(f"{i}. {name} (зарплата: {salary_str})\n")
        
        print(f"Отчет сохранен в файле {report_file}")
        
    except Exception as e:
        print(f"ошибка, не удалось сохранить отчет: {e}")

#def quick_test():
#    return test_full_pipeline(pages=2)

def full_test():
    return test_full_pipeline(pages=10)

if __name__ == "__main__":

    success = full_test()
    if success:
        print("Все работает корректно")
    else:
        print("Обнаружены проблемы в пайплане")