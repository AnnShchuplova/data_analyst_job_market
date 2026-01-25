import sys
import os
import unittest
import pandas as pd
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.hh_parser import HHParser
from src.data_collection.filters import remove_duplicates 
from src.data_collection.filters import filter_data_analyst_vacancies
from src.data_processing.cleaner import DataCleaner
        

class TestHHParser(unittest.TestCase): 
    
    def test_parser_initialization(self):
        
        parser = HHParser(delay=0.1)
        self.assertEqual(parser.delay, 0.1)
        print("HHParser инициализируется корректно")
    
    def test_is_analyst_by_role(self):
        parser = HHParser()
        
        analyst_vac = {"professional_roles": [{"id": "156"}]}
        self.assertTrue(parser.is_analyst_by_role(analyst_vac))
        
        non_analyst_vac = {"professional_roles": [{"id": "1"}]}
        self.assertFalse(parser.is_analyst_by_role(non_analyst_vac))
        
        print("Фильтрация по ролям работает")

class TestFilters(unittest.TestCase): 
    
    def test_filter_analyst(self):
        vacancies = [
            {
                "id": 1, 
                "name": "Аналитик данных",
                "professional_roles": [{"id": "156", "name": "Аналитик данных"}]
            },
            {
                "id": 2, 
                "name": "Разработчик Python",
                "professional_roles": [{"id": "1", "name": "Программист"}]
            },
            {
                "id": 3,
                "name": "Data Analyst",
                "professional_roles": [{"id": "156", "name": "Аналитик данных"}]
            }
        ]
        
        filtered = filter_data_analyst_vacancies(vacancies)
        filtered_ids = {vac["id"] for vac in filtered}
        expected_ids = {1, 3}
        
        self.assertEqual(filtered_ids, expected_ids, 
                        f"Ожидалось {expected_ids}, получили {filtered_ids}")
        print("Фильтр аналитиков работает корректно")
    
    def test_remove_duplicates(self):
        vacancies = [
            {"id": 1, "name": "Вакансия 1"},
            {"id": 2, "name": "Вакансия 2"},
            {"id": 1, "name": "Вакансия 1"}, 
            {"id": 3, "name": "Вакансия 3"},
        ]
        
        unique = remove_duplicates(vacancies)
        
        self.assertEqual(len(unique), 3) 
        unique_ids = {v["id"] for v in unique}
        self.assertEqual(unique_ids, {1, 2, 3})
        print("Удаление дубликатов работает")

class TestCleaner(unittest.TestCase):
    def test_calculate_avg_salary(self):
        cleaner = DataCleaner()
        
        result = cleaner._calculate_avg_salary(100000, 150000)
        self.assertEqual(result, 125000, f"Ожидалось 125000, получили {result}")
        
        result = cleaner._calculate_avg_salary(100000, None)
        self.assertEqual(result, 100000, f"Ожидалось 100000, получили {result}")
        
        result = cleaner._calculate_avg_salary(None, 150000)
        self.assertEqual(result, 150000, f"Ожидалось 150000, получили {result}")
       
        result = cleaner._calculate_avg_salary(None, None)
        self.assertTrue(np.isnan(result), f"Ожидалось NaN, получили {result}")
        
        print("Расчет средней зарплаты работает корректно")
    
    def test_clean_experience(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "experience": [
                {"id": "noExperience", "name": "Нет опыта"},
                {"id": "between1And3", "name": "От 1 года до 3 лет"},
                {"id": "between3And6", "name": "От 3 до 6 лет"},
                None 
            ]
        })
        
        cleaned = cleaner.clean_experience(df)
        
        self.assertIn("experience_id", cleaned.columns)
        self.assertIn("experience_name", cleaned.columns)
        self.assertIn("estimated_experience_years", cleaned.columns)
        
        print("Очистка опыта работает корректно")

def run_tests():

    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTests(loader.loadTestsFromTestCase(TestHHParser))
    test_suite.addTests(loader.loadTestsFromTestCase(TestFilters))
    test_suite.addTests(loader.loadTestsFromTestCase(TestCleaner))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    
    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    passed = total - failed - errors
    
    if failed == 0 and errors == 0:
        print("Тесты пройдены успешно")
    else:
        for test, traceback in result.failures:
            print(f"Провален: {test}")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"Ошибка: {test}")
            print(traceback)
    
    return result

if __name__ == "__main__":
    run_tests()