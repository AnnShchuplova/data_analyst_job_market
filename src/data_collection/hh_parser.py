import requests
import time
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HHParser:
    base_url = "https://api.hh.ru/vacancies"
    
    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.session = requests.Session()
        
    def fetch_vacancies(self, 
                       query: str = "аналитик",
                       area: int = 113,
                       experience: str = None,
                       pages: int = 20,
                       only_with_salary: bool = True,
                       professional_role: str = None) -> list: 
        
        vacancies = []
        
        for page in range(pages):
            params = {
                "text": query,
                "area": area,
                "page": page,
                "per_page": 100,
                "experience": experience,
                "only_with_salary": only_with_salary
            }

            if professional_role:
                params["professional_role"] = professional_role

            try:
                response = self.session.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "items" not in data or not data['items']:
                    logger.info(f"На странице {page} нет вакансий")
                    break
                
                filtered_items = []
                for item in data["items"]:
                    if self.is_analyst_by_role(item):
                        filtered_items.append(item)
                
                vacancies.extend(filtered_items)
                logger.info(f"На странице {page} собрано {len(filtered_items)} аналитических вакансий")
                
                time.sleep(self.delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка при запросе страницы {page}: {e}")
                break
        
        logger.info(f"Всего собрано {len(vacancies)} аналитических вакансий")
        return vacancies
    
    def is_analyst_by_role(self, vacancy: dict) -> bool:
        if "professional_roles" not in vacancy:
            return False
            
        roles = vacancy["professional_roles"]
        if not roles:
            return False
        
        role_ids = [str(role.get("id")) for role in roles]
        
        analyst_role_ids = ["10", "134", "148", "150", "156", "163", "164"]
        has_analyst_role = any(role_id in analyst_role_ids for role_id in role_ids)
        
        return has_analyst_role
    
    def save_raw_data(self, vacancies: list, filename: str = "data/raw/vacancies.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
        logger.info(f"Сырые данные сохранены в {filename}")