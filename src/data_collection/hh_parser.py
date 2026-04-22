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
        
    def _make_request_with_retry(self, params: dict, max_retries: int = 3) -> requests.Response:
        """Выполнение HTTP-запроса
        
        Корректно обрабатывает таймауты, отсутствие интернета,
        недоступность API с повторными попытками.
        """
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Таймаут при запросе (попытка {attempt}/{max_retries}): {e}")
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Ошибка соединения (попытка {attempt}/{max_retries}): {e}")
            except requests.exceptions.HTTPError as e:
                # Для 4xx ошибок не имеет смысла повторять
                if response is not None and 400 <= response.status_code < 500:
                    raise
                last_exception = e
                logger.warning(f"HTTP ошибка (попытка {attempt}/{max_retries}): {e}")
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Ошибка запроса (попытка {attempt}/{max_retries}): {e}")
            
            if attempt < max_retries:
                wait_time = 2 ** attempt  # экспоненциальная задержка: 2, 4 сек
                logger.info(f"Повторная попытка через {wait_time} сек...")
                time.sleep(wait_time)
        
        logger.error(f"Все {max_retries} попытки запроса завершились неудачей")
        raise requests.exceptions.RequestException(
            f"Не удалось выполнить запрос после {max_retries} попыток"
        ) from last_exception

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
                response = self._make_request_with_retry(params)
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
                logger.error(f"Ошибка при запросе страницы {page} после всех попыток: {e}")
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